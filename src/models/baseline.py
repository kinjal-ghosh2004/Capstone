import torch
import numpy as np
import sys
import os

# allow imports from parent directory (src)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoModelForSequenceClassification
from data.loaders import load_thar, load_indic_sentiment
from data.tokenize_utils import get_tokenizer, tokenize_dataset, create_dataloader, MODEL_NAME
import sys
import os

def evaluate_baseline(model, dataloader, device):
    """
    Evaluates the model without any training to get the baseline zero-shot/random performance.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    print(f"Evaluating on {len(dataloader.dataset)} samples...")
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(all_labels, all_preds, zero_division=0)
    
    return acc, f1, report

def run_baselines():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = get_tokenizer()
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # We will test on THAR (Hate Speech - binary) and IndicSentiment (Sentiment - 3 classes)
    
    # 1. THAR Hate Speech Baseline
    print("\n--- Hate Speech (THAR) Baseline ---")
    thar_path = os.path.join(base_dir, 'Datasets', 'Hate Speech Training', 'THAR', 'THAR-Dataset.csv')
    thar_ds = load_thar(thar_path)
    thar_ds = thar_ds.select(range(500)) # take a subset for quick baseline
    
    thar_tokenized = tokenize_dataset(thar_ds, tokenizer, max_length=128)
    thar_loader = create_dataloader(thar_tokenized, batch_size=16, shuffle=False)
    
    model_hate = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model_hate.to(device)
    
    hate_acc, hate_f1, hate_report = evaluate_baseline(model_hate, thar_loader, device)
    print(f"Accuracy: {hate_acc:.4f}, F1 (Macro): {hate_f1:.4f}")
    
    # 2. Sentiment Baseline
    print("\n--- Sentiment (IndicSentiment Hindi) Baseline ---")
    sent_path = os.path.join(base_dir, 'Datasets', 'Sentiment Training', 'IndicSentiment', 'data', 'validation', 'hi.json')
    sent_ds = load_indic_sentiment(sent_path)
    sent_tokenized = tokenize_dataset(sent_ds, tokenizer, max_length=128)
    sent_loader = create_dataloader(sent_tokenized, batch_size=16, shuffle=False)
    
    model_sent = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    model_sent.to(device)
    
    sent_acc, sent_f1, sent_report = evaluate_baseline(model_sent, sent_loader, device)
    print(f"Accuracy: {sent_acc:.4f}, F1 (Macro): {sent_f1:.4f}")
    
    # Save the output to results folder
    out_dir = os.path.join(base_dir, 'results')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'baseline_metrics.txt'), 'w') as f:
        f.write("=== HATE SPEECH BASELINE (THAR) ===\n")
        f.write(f"Accuracy: {hate_acc:.4f}\n")
        f.write(f"F1 (Macro): {hate_f1:.4f}\n")
        f.write("Classification Report:\n")
        f.write(hate_report)
        f.write("\n\n")
        
        f.write("=== SENTIMENT BASELINE (Hindi) ===\n")
        f.write(f"Accuracy: {sent_acc:.4f}\n")
        f.write(f"F1 (Macro): {sent_f1:.4f}\n")
        f.write("Classification Report:\n")
        f.write(sent_report)
        
    print("\nBaseline metrics saved to results/baseline_metrics.txt")

if __name__ == "__main__":
    run_baselines()
