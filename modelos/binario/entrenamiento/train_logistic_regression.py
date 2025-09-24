import pickle
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Configuración de MLflow
mlflow.set_experiment("BiomedicalClassifierComparison")

# Cargar datos
with open("train_data.pkl", "rb") as f:
    X_train_sparse, y_train = pickle.load(f)
with open("val_data.pkl", "rb") as f:
    X_val_sparse, y_val = pickle.load(f)
with open("test_data.pkl", "rb") as f:
    X_test_sparse, y_test = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Entrenar Logistic Regression
model_name = "LogisticRegression"
print(f"Entrenando {model_name}...")

with mlflow.start_run(run_name=f"Sparse - {model_name}"):
    mlflow.log_param("model_type", "sparse")
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("vectorizer", "TF-IDF")
    mlflow.log_param("train_size", X_train_sparse.shape[0])
    mlflow.log_param("val_size", X_val_sparse.shape[0])
    mlflow.log_param("test_size", X_test_sparse.shape[0])
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    params = {"C": [0.1, 1, 10]}
    cv_folds = min(3, X_train_sparse.shape[0] // 2)
    if cv_folds < 2:
        cv_folds = 2
    
    grid_search = GridSearchCV(model, params, cv=cv_folds, scoring='f1', n_jobs=1)
    
    try:
        grid_search.fit(X_train_sparse, y_train)
        best_model = grid_search.best_estimator_
        
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        
        y_val_pred = best_model.predict(X_val_sparse)
        val_precision = precision_score(y_val, y_val_pred, average='binary', zero_division=0)
        val_recall = recall_score(y_val, y_val_pred, average='binary', zero_division=0)
        val_f1 = f1_score(y_val, y_val_pred, average='binary', zero_division=0)
        
        mlflow.log_metric("val_precision", val_precision)
        mlflow.log_metric("val_recall", val_recall)
        mlflow.log_metric("val_f1", val_f1)
        
        y_test_pred = best_model.predict(X_test_sparse)
        test_precision = precision_score(y_test, y_test_pred, average='binary', zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, average='binary', zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, average='binary', zero_division=0)
        
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1", test_f1)
        
        # Guardar modelo en MLflow y en disco
        mlflow.sklearn.log_model(best_model, "model")
        mlflow.sklearn.log_model(vectorizer, "vectorizer")
        with open("logistic_regression_model.pkl", "wb") as f:
            pickle.dump(best_model, f)
        with open("logistic_regression_val_f1.txt", "w") as f:
            f.write(str(val_f1))
        
        print(f"  Val F1: {val_f1:.3f}, Test F1: {test_f1:.3f}")
        print(f"Modelo guardado como logistic_regression_model.pkl")
        
    except Exception as e:
        print(f"Error entrenando {model_name}: {e}")