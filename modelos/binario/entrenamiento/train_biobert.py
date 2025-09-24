import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import mlflow
import mlflow.pytorch
from datasets import load_from_disk
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import shutil
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Configuración de MLflow - usar experimento existente
mlflow.set_experiment("BiomedicalClassifierComparison")

# Cerrar cualquier run activo para evitar conflictos
if mlflow.active_run():
    mlflow.end_run()

# Cargar datasets
print("Cargando datasets...")
try:
    train_dataset = torch.load("train_dataset.pt")
    val_dataset = torch.load("val_dataset.pt")
    test_dataset = torch.load("test_dataset.pt")
    print(f"Datasets cargados - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
except Exception as e:
    print(f"Error cargando datasets: {e}")
    exit(1)

# Entrenar BioBERT
model_name = "dmis-lab/biobert-base-cased-v1.1"
print("Entrenando BioBERT...")

# Crear un nuevo run con nombre único
run_name = f"Contextual-BioBERT-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

with mlflow.start_run(run_name=run_name):
    mlflow.log_param("model_type", "contextual")
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("train_size", len(train_dataset))
    mlflow.log_param("val_size", len(val_dataset))
    mlflow.log_param("test_size", len(test_dataset))
    mlflow.log_param("framework", "transformers")
    mlflow.log_param("task", "binary_classification")
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        # CLAVE: Deshabilitar completamente la integración automática de MLflow
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            remove_unused_columns=False,
            seed=42,
            data_seed=42,
            fp16=False,
            dataloader_drop_last=False,
            report_to=[],  # Lista vacía para deshabilitar auto-logging
            disable_tqdm=False,
        )
        
        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            precision = precision_score(labels, preds, average='binary', zero_division=0)
            recall = recall_score(labels, preds, average='binary', zero_division=0)
            f1 = f1_score(labels, preds, average='binary', zero_division=0)
            return {"precision": precision, "recall": recall, "f1": f1}
        
        # Agregar parámetros de entrenamiento a MLflow
        mlflow.log_param("epochs", 5)
        mlflow.log_param("batch_size", 8)
        mlflow.log_param("warmup_steps", 100)
        mlflow.log_param("weight_decay", 0.01)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        
        print("Iniciando entrenamiento...")
        trainer.train()
        
        print("Evaluando en conjunto de validación...")
        val_results = trainer.evaluate(val_dataset)
        mlflow.log_metric("val_precision", val_results['eval_precision'])
        mlflow.log_metric("val_recall", val_results['eval_recall'])
        mlflow.log_metric("val_f1", val_results['eval_f1'])
        mlflow.log_metric("val_loss", val_results['eval_loss'])
        
        print("Evaluando en conjunto de prueba...")
        test_results = trainer.evaluate(test_dataset)
        mlflow.log_metric("test_precision", test_results['eval_precision'])
        mlflow.log_metric("test_recall", test_results['eval_recall'])
        mlflow.log_metric("test_f1", test_results['eval_f1'])
        mlflow.log_metric("test_loss", test_results['eval_loss'])
        
        # Guardar modelo en MLflow y en disco
        print("Guardando modelo...")
        mlflow.pytorch.log_model(model, "model")
        trainer.save_model("./biobert_model")
        
        # Guardar F1 scores para el ensemble (IMPORTANTE para que funcione el ensemble)
        with open("biobert_val_f1.txt", "w") as f:
            f.write(str(val_results['eval_f1']))
        with open("biobert_test_f1.txt", "w") as f:
            f.write(str(test_results['eval_f1']))
        
        # Log de archivos adicionales
        mlflow.log_artifact("biobert_val_f1.txt")
        mlflow.log_artifact("biobert_test_f1.txt")
        
        print(f"\n=== RESULTADOS BIOBERT ===")
        print(f"Validación - Precisión: {val_results['eval_precision']:.3f}, Recall: {val_results['eval_recall']:.3f}, F1: {val_results['eval_f1']:.3f}")
        print(f"Prueba - Precisión: {test_results['eval_precision']:.3f}, Recall: {test_results['eval_recall']:.3f}, F1: {test_results['eval_f1']:.3f}")
        print(f"Modelo guardado en './biobert_model'")
        print(f"Archivos F1 generados: biobert_val_f1.txt, biobert_test_f1.txt")
        
    except Exception as e:
        print(f"Error entrenando BioBERT: {e}")
        import traceback
        traceback.print_exc()
        mlflow.log_param("status", "failed")
        mlflow.log_param("error", str(e))
    
    finally:
        # Limpiar archivos temporales
        try:
            if os.path.exists("./results"):
                shutil.rmtree("./results")
            if os.path.exists("./logs"):
                shutil.rmtree("./logs")
        except Exception as e:
            print(f"Error limpiando archivos temporales: {e}")

print("Script completado.")