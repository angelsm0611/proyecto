import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import evaluate
import mlflow
import torch
from textstat import flesch_kincaid_grade
import os

data_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(data_dir,r"data\data.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

records = []

for _, articles in data.items():
    if isinstance(articles, dict):  # aseguramos que es un diccionario
        for _, content in articles.items():
            if isinstance(content, dict):  # solo seguimos si es un dict
                # Unir abstract si existe
                abstract = ""
                if "abstract" in content and isinstance(content["abstract"], dict):
                    abstract = " ".join(content["abstract"].values())

                # Unir adaptation si existe
                adaptation = ""
                if "adaptations" in content and isinstance(content["adaptations"], dict):
                    adaptation_dict = content["adaptations"].get("adaptation2", {})
                    if isinstance(adaptation_dict, dict):
                        adaptation = " ".join(adaptation_dict.values())

                if abstract and adaptation:  # solo guardamos pares válidos
                    records.append({"source": abstract, "target": adaptation})

# Convertimos a DataFrame
import pandas as pd
df = pd.DataFrame(records)
print(df.head())
total_ejemplos = len(df)
print(f"Total de ejemplos: {total_ejemplos}")
dataset = Dataset.from_pandas(df)


small_models = [
    "distilgpt2",
    "gpt2",
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/pythia-70m"
]


def tokenize_function(examples, tokenizer):
    inputs = [f"Simplify: {src}" for src in examples["source"]]
    targets = examples["target"]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length").input_ids
    model_inputs["labels"] = labels
    return model_inputs

def train_model(model_name, dataset):
    print(f"\n🔧 Entrenando {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    model = AutoModelForCausalLM.from_pretrained(model_name)

    if "pythia" in model_name:
        target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    elif "gpt2" in model_name or "ettin" in model_name:
         target_modules = ["c_attn","q_proj","v_proj"]
    else:
         target_modules = None 

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    output_dir = f".traductor/checkpoint/{model_name.replace('/', '_')}"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,   
        weight_decay=0.01,
        save_total_limit=1,
        fp16=True,
        push_to_hub=False,
        logging_dir="./logs",
        report_to="mlflow",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized.shuffle(seed=42).select(range(total_ejemplos)),
        eval_dataset=tokenized.shuffle(seed=42).select(range(100)),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Modelo {model_name} guardado en {output_dir}")
    return model, tokenizer


trained_models = {}
for m in small_models:
    model, tokenizer = train_model(m, dataset)
    trained_models[m] = (model, tokenizer)


bertscore = evaluate.load("bertscore")

def evaluate_model(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = inputs.to(model.device) 
    outputs = model.generate(**inputs, max_new_tokens=100)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)


    refs = [df["target"][0]]
    preds = [summary]
    bs = bertscore.compute(predictions=preds, references=refs, lang="en")


    fk = flesch_kincaid_grade(summary)

    return summary, bs["f1"][0], fk

sample_text = df["source"][0]
for m, (model, tok) in trained_models.items():
    summary, bert, fk = evaluate_model(model, tok, sample_text)
    print(f"\n Modelo: {m}")
    print(f" Resumen: {summary}")
    print(f" BERTScore F1: {bert:.3f} | Flesch-Kincaid: {fk:.2f}")