import pickle
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.nn.functional import softmax
import os
import warnings
import shutil
warnings.filterwarnings('ignore')

# Configuración de MLflow
mlflow.set_experiment("BiomedicalClassifierComparison")

# Cargar datos
with open("test_data.pkl", "rb") as f:
    X_test_sparse, y_test = pickle.load(f)
test_dataset = torch.load("test_dataset.pt")

# Encontrar el mejor modelo disperso
sparse_models = ["logistic_regression", "svm", "random_forest", "naive_bayes"]
best_f1_val = -1
best_sparse_name = None
best_sparse_model = None

for model_name in sparse_models:
    try:
        with open(f"{model_name}_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(f"{model_name}_val_f1.txt", "r") as f:
            f1_val = float(f.read())
        if f1_val > best_f1_val:
            best_f1_val = f1_val
            best_sparse_name = model_name
            best_sparse_model = model
    except FileNotFoundError:
        print(f"Modelo {model_name} no encontrado, omitiendo...")
        continue

# Cargar resultados de BioBERT
contextual_trained = False
try:
    with open("biobert_val_f1.txt", "r") as f:
        contextual_val_f1 = float(f.read())
    with open("biobert_test_f1.txt", "r") as f:
        contextual_test_f1 = float(f.read())
    contextual_model = AutoModelForSequenceClassification.from_pretrained("./biobert_model")
    contextual_trained = True
except Exception as e:
    print(f"Error cargando BioBERT: {e}")
    contextual_trained = False

# Entrenar ensamble
if contextual_trained and best_sparse_model:
    print("\n=== Entrenando ensamble ===")
    with mlflow.start_run(run_name="Ensemble - Best Sparse + Contextual"):
        mlflow.log_param("model_type", "ensemble")
        mlflow.log_param("components", f"{best_sparse_name} + BioBERT")
        mlflow.log_param("dataset_size", len(y_test))
        mlflow.log_param("sparse_model_f1", best_f1_val)
        mlflow.log_param("contextual_model_f1", contextual_val_f1)
        
        try:
            # Obtener probabilidades del mejor modelo disperso
            sparse_probs = best_sparse_model.predict_proba(X_test_sparse)[:, 1]
            
            # Obtener probabilidades de BioBERT
            training_args = TrainingArguments(
                output_dir="./results",
                per_device_eval_batch_size=8,
                remove_unused_columns=False,
                report_to=None,
            )
            trainer = Trainer(
                model=contextual_model,
                args=training_args,
                eval_dataset=test_dataset,
            )
            contextual_predictions = trainer.predict(test_dataset)
            contextual_logits = contextual_predictions.predictions
            contextual_probs = softmax(torch.tensor(contextual_logits), dim=-1)[:, 1].numpy()
            
            # Combinar con pesos basados en F1 de validación
            sparse_weight = best_f1_val
            contextual_weight = contextual_val_f1
            total_weight = sparse_weight + contextual_weight
            sparse_norm_weight = sparse_weight / total_weight
            contextual_norm_weight = contextual_weight / total_weight
            
            mlflow.log_param("sparse_weight", sparse_norm_weight)
            mlflow.log_param("contextual_weight", contextual_norm_weight)
            
            ensemble_probs = (sparse_norm_weight * sparse_probs + 
                            contextual_norm_weight * contextual_probs)
            ensemble_preds = (ensemble_probs > 0.5).astype(int)
            
            precision = precision_score(y_test, ensemble_preds, average='binary', zero_division=0)
            recall = recall_score(y_test, ensemble_preds, average='binary', zero_division=0)
            f1 = f1_score(y_test, ensemble_preds, average='binary', zero_division=0)
            
            mlflow.log_metric("test_precision", precision)
            mlflow.log_metric("test_recall", recall)
            mlflow.log_metric("test_f1", f1)
            
            best_individual_test_f1 = max(
                f1_score(y_test, best_sparse_model.predict(X_test_sparse), average='binary', zero_division=0),
                contextual_test_f1
            )
            mlflow.log_metric("improvement_over_best_individual", f1 - best_individual_test_f1)
            
            ensemble_description = f"""
            Ensemble Model Description:
            - Component 1: {best_sparse_name} (weight: {sparse_norm_weight:.3f}, F1: {best_f1_val:.3f})
            - Component 2: BioBERT (weight: {contextual_norm_weight:.3f}, F1: {contextual_val_f1:.3f})
            - Combination method: Performance-weighted soft voting
            - Final Test F1 Score: {f1:.3f}
            - Best individual F1: {best_individual_test_f1:.3f}
            - Improvement over best individual: {f1 - best_individual_test_f1:.3f}
            """
            mlflow.log_text(ensemble_description, "model_description.txt")
            
            print(f"Ensamble - Test F1: {f1:.3f}")
            print(f"Mejora sobre mejor individual: {f1 - best_individual_test_f1:.3f}")
            
        except Exception as e:
            print(f"Error en ensamble: {e}")
            import traceback
            traceback.print_exc()
else:
    print("Ensamble no ejecutado: falta modelo disperso o contextual")

# Limpiar archivos temporales
if os.path.exists("./results"):
    shutil.rmtree("./results")