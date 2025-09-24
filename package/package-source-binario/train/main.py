import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
import torch
from datasets import Dataset
import pickle
import os
import subprocess
import warnings


warnings.filterwarnings('ignore')

# Cargar datos
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construir la ruta absoluta a dataset_binario.json en train/data/
data_path = os.path.join(script_dir, "data", "dataset_binario.csv")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"No se encontró el archivo en: {data_path}")
df = pd.read_csv(data_path)

print(f"Dataset shape: {df.shape}")
print(f"Label distribution:\n{df['label'].value_counts()}")

# Limpieza mínima: convertir a minúsculas
df['text'] = df['text'].str.lower()

# División train/validation/test
test_size = max(0.2, 2/len(df))
val_size = max(0.2, 2/(len(df)*(1-test_size)))

try:
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=42, 
        stratify=df['label']
    )
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=val_size, 
        random_state=42, 
        stratify=train_val_df['label']
    )
except ValueError as e:
    print(f"Error en estratificación: {e}")
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size, random_state=42)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Preparar datos para modelos dispersos
max_features = min(1000, len(train_df) * 10)
vectorizer = TfidfVectorizer(
    max_features=max_features,
    min_df=1,
    ngram_range=(1, 2)
)

try:
    X_train_sparse = vectorizer.fit_transform(train_df['text'])
    X_val_sparse = vectorizer.transform(val_df['text'])
    X_test_sparse = vectorizer.transform(test_df['text'])
    
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    print(f"Feature matrix shape: {X_train_sparse.shape}")

    models_dir = os.path.join(script_dir, "..", "clasificador", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Guardar datos dispersos
    with open("clasificador/models/train_data.pkl", "wb") as f:
        pickle.dump((X_train_sparse, y_train), f)
    with open("clasificador/models/val_data.pkl", "wb") as f:
        pickle.dump((X_val_sparse, y_val), f)
    with open("clasificador/models/test_data.pkl", "wb") as f:
        pickle.dump((X_test_sparse, y_test), f)
    with open("clasificador/models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
except Exception as e:
    print(f"Error en vectorización: {e}")
    raise

# Preparar datos para modelo contextual
model_name = "dmis-lab/biobert-base-cased-v1.1"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(texts):
        return tokenizer(
            texts.tolist(), 
            padding="max_length", 
            truncation=True, 
            max_length=128,
            return_tensors='pt'
        )
    
    train_tokenized = tokenize_function(train_df['text'])
    train_tokenized['labels'] = torch.tensor(y_train, dtype=torch.long)
    train_dataset = Dataset.from_dict({
        'input_ids': train_tokenized['input_ids'],
        'attention_mask': train_tokenized['attention_mask'],
        'labels': train_tokenized['labels']
    })
    
    val_tokenized = tokenize_function(val_df['text'])
    val_tokenized['labels'] = torch.tensor(y_val, dtype=torch.long)
    val_dataset = Dataset.from_dict({
        'input_ids': val_tokenized['input_ids'],
        'attention_mask': val_tokenized['attention_mask'],
        'labels': val_tokenized['labels']
    })
    
    test_tokenized = tokenize_function(test_df['text'])
    test_tokenized['labels'] = torch.tensor(y_test, dtype=torch.long)
    test_dataset = Dataset.from_dict({
        'input_ids': test_tokenized['input_ids'],
        'attention_mask': test_tokenized['attention_mask'],
        'labels': test_tokenized['labels']
    })
    
    # Guardar datasets para BioBERT

    torch.save(train_tokenized, "clasificador/models/train_dataset.pt")
    torch.save(val_tokenized, "clasificador/models/val_dataset.pt")
    torch.save(test_tokenized, "clasificador/models/test_dataset.pt")
    print("Datasets tokenizados y guardados para BioBERT.")
except Exception as e:
    print(f"Error en tokenización: {e}")
    print("Continuando sin datos contextuales...")


def run_training_script(script_name):
    print(f"\n=== Ejecutando {script_name} ===")
    try:
        result = subprocess.run(['python', script_name], check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Advertencias/Errores en {script_name}:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error ejecutando {script_name}: {e}")
        print(f"Salida de error: {e.stderr}")

def main():
    # Paso 1: Preparar datos
    #prepare_data()

    # Paso 2: Entrenar modelos dispersos y BioBERT
    scripts = [
        "train/train_logistic_regression.py",
        "train/train_svm.py",
        "train/train_random_forest.py",
        "train/train_naive_bayes.py",
        "train/train_biobert.py"
    ]
    
    for script in scripts:
        if os.path.exists(script):
            run_training_script(script)
        else:
            print(f"Script {script} no encontrado, omitiendo...")

    # Paso 3: Entrenar ensamble
    if os.path.exists("train/train_ensemble.py"):
        print("\n=== Ejecutando train_ensemble.py ===")
        run_training_script("train/train_ensemble.py")
    else:
        print("Script train_ensemble.py no encontrado, omitiendo...")

    print("\n=== Ejecución completa ===")

if __name__ == "__main__":
    main()