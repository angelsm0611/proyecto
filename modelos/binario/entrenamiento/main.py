import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
import torch
from datasets import Dataset
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Cargar datos
df = pd.read_csv("C:\\Users\\sanch\\Documents\\proyecto grado\\data\\dataset_binario.csv")

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
    
    # Guardar datos dispersos
    with open("train_data.pkl", "wb") as f:
        pickle.dump((X_train_sparse, y_train), f)
    with open("val_data.pkl", "wb") as f:
        pickle.dump((X_val_sparse, y_val), f)
    with open("test_data.pkl", "wb") as f:
        pickle.dump((X_test_sparse, y_test), f)
    with open("vectorizer.pkl", "wb") as f:
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
    torch.save(train_dataset, "train_dataset.pt")
    torch.save(val_dataset, "val_dataset.pt")
    torch.save(test_dataset, "test_dataset.pt")
except Exception as e:
    print(f"Error en tokenización: {e}")
    print("Continuando sin datos contextuales...")