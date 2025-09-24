import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

def load_model_and_tokenizer(checkpoint_dir, base_model_name=None):
    """
    Load the fine-tuned model and tokenizer from the checkpoint directory.
    
    Args:
        checkpoint_dir (str): Path to the fine-tuned model checkpoint.
        base_model_name (str, optional): Name of the base model (e.g., 'distilgpt2').
            If None, inferred from checkpoint_dir.
    
    Returns:
        tuple: (model, tokenizer, device)
    """
    # Infer base model name from checkpoint_dir if not provided
    if base_model_name is None:
        base_model_name = checkpoint_dir.split("/")[-1].replace("_", "/")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model and apply fine-tuned weights
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    
    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device

def predict_(text, checkpoint_dir, base_model_name=None):
    """
    Generate a simplified version of the input text using the fine-tuned model.
    
    Args:
        text (str): Input text to simplify.
        checkpoint_dir (str): Path to the fine-tuned model checkpoint.
        base_model_name (str, optional): Name of the base model (e.g., 'distilgpt2').
    
    Returns:
        str: Simplified text.
    """
    model, tokenizer, device = load_model_and_tokenizer(checkpoint_dir, base_model_name)
    
    # Preprocess input
    input_text = f"Simplify: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True, padding="max_length")
    inputs = inputs.to(device)
    
    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

if __name__ == "__main__":
    # Example usage
    sample_text = "The rapid advancement of artificial intelligence technologies has significantly impacted various industries."
    checkpoint_dir = "./checkpoints/distilgpt2"  # Adjust to your best model's checkpoint
    simplified_text = predict_(sample_text, checkpoint_dir)
    print(f"Input: {sample_text}")
    print(f"Simplified Output: {simplified_text}")