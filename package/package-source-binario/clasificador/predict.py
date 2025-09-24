import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.nn.functional import softmax
from datasets import Dataset
import numpy as np
import os


def load_sparse_model(model_name, model_path="clasificador/models"):
    try:
        model_file = os.path.join(model_path, f"{model_name}_model.pkl")
        vectorizer_file = os.path.join(model_path, "vectorizer.pkl")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file {model_file} not found")
        if not os.path.exists(vectorizer_file):
            raise FileNotFoundError(f"Vectorizer file {vectorizer_file} not found")
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        with open(vectorizer_file, "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError as e:
        print(f"Error loading {model_name}: {e}")
        return None, None

def load_biobert_model(model_path="clasificador/models/biobert_model", model_name="dmis-lab/biobert-base-cased-v1.1"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading BioBERT: {e}")
        return None, None

def predict_sparse(text, model, vectorizer):
    if model is None or vectorizer is None:
        return None, None
    text = text.lower()
    X = vectorizer.transform([text])
    label = model.predict(X)[0]
    prob = model.predict_proba(X)[0, 1]
    return label, prob

def predict_biobert(text, model, tokenizer):
    if model is None or tokenizer is None:
        return None, None
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    training_args = TrainingArguments(
        output_dir="./temp_results",
        per_device_eval_batch_size=1,
        report_to=[],
    )
    trainer = Trainer(model=model, args=training_args)
    dataset = Dataset.from_dict({
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
    })
    predictions = trainer.predict(dataset)
    probs = softmax(torch.tensor(predictions.predictions), dim=-1)[:, 1].numpy()[0]
    label = (probs > 0.5).astype(int)
    return label, probs

def predict_ensemble(text, sparse_model_name, sparse_model, vectorizer, biobert_model, biobert_tokenizer, sparse_f1, biobert_f1):
    if sparse_model is None or biobert_model is None:
        return None, None
    sparse_label, sparse_prob = predict_sparse(text, sparse_model, vectorizer)
    biobert_label, biobert_prob = predict_biobert(text, biobert_model, biobert_tokenizer)
    if sparse_prob is None or biobert_prob is None:
        return None, None
    total_weight = sparse_f1 + biobert_f1
    sparse_weight = sparse_f1 / total_weight
    biobert_weight = biobert_f1 / total_weight
    ensemble_prob = sparse_weight * sparse_prob + biobert_weight * biobert_prob
    ensemble_label = (ensemble_prob > 0.5).astype(int)
    return ensemble_label, ensemble_prob

def predict(text, sparse_model_name="logistic_regression", model_path="clasificador/models", sparse_f1_path=None, biobert_f1_path="clasificador/models/biobert_val_f1.txt"):
    sparse_model, vectorizer = load_sparse_model(sparse_model_name, model_path)
    biobert_model, biobert_tokenizer = load_biobert_model(os.path.join(model_path, "biobert_model"))
    sparse_f1 = 0.0
    biobert_f1 = 0.0
    if sparse_f1_path and os.path.exists(sparse_f1_path):
        with open(sparse_f1_path, "r") as f:
            sparse_f1 = float(f.read())
    if biobert_f1_path and os.path.exists(biobert_f1_path):
        with open(biobert_f1_path, "r") as f:
            biobert_f1 = float(f.read())
    if sparse_model and biobert_model and sparse_f1 > 0 and biobert_f1 > 0:
        label, prob = predict_ensemble(text, sparse_model_name, sparse_model, vectorizer, biobert_model, biobert_tokenizer, sparse_f1, biobert_f1)
        return label, prob, "ensemble"
    if sparse_model:
        label, prob = predict_sparse(text, sparse_model, vectorizer)
        return label, prob, sparse_model_name
    if biobert_model:
        label, prob = predict_biobert(text, biobert_model, biobert_tokenizer)
        return label, prob, "biobert"
    raise ValueError("No valid model available for prediction")