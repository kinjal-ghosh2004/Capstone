import os
import argparse
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import ClassLabel
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.loaders import load_thar, load_indic_sentiment
from data.tokenize_utils import get_tokenizer, tokenize_dataset

def plot_confusion_matrix(y_true, y_pred, classes, title, output_path, labels_list):
    cm = confusion_matrix(y_true, y_pred, labels=labels_list)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved confusion matrix to {output_path}")

def plot_comparative_analysis(task, our_f1, output_path):
    # Hardcoded baselines from Phase 6 Documentation
    if task == 'hate':
        labels = ['Zero-Shot', 'Standard Fine-Tuning', 'DAPT + Optimized (Ours)']
        f1_scores = [30.3, 77.5, our_f1 * 100]
        title = "Hate Speech Detection: Macro F1 Comparison"
    else:
        labels = ['Zero-Shot', 'Standard Fine-Tuning', 'DAPT + Optimized (Ours)']
        f1_scores = [49.3, 36.0, our_f1 * 100]
        title = "Sentiment Analysis: Macro F1 Comparison"

    plt.figure(figsize=(8, 6))
    colors = ['#d3d3d3', '#a9a9a9', '#4c72b0']
    bars = plt.bar(labels, f1_scores, color=colors)
    plt.ylabel('Macro F1 Score (%)')
    plt.title(title)
    plt.ylim(0, 100)

    # Add text on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved comparative analysis chart to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-Tuned IndicBERT")
    parser.add_argument('--task', type=str, required=True, choices=['hate', 'sentiment'], help="Task to evaluate")
    args = parser.parse_args()
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_path = os.path.join(base_dir, 'results', 'models', args.task, 'best_model')
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find model at {model_path}. Have you trained it yet?")
        sys.exit(1)
        
    print(f"Loading best model from {model_path}...")
    tokenizer = get_tokenizer(model_path)
    
    # Load and prepare exact same dataset
    from datasets import concatenate_datasets
    
    if args.task == 'hate':
        ds_list = []
        thar_path = os.path.join(base_dir, 'Datasets', 'Hate Speech Training', 'THAR', 'THAR-Dataset.csv')
        thar_ds = load_thar(thar_path)
        if thar_ds: ds_list.append(thar_ds)
        
        from data.loaders import load_all_dravidian
        dravidian_base = os.path.join(base_dir, 'Datasets', 'Hate Speech Training', 'Dravidian-Offensive-Language-Identification', 'Datasets')
        for split in ['train', 'dev']:
            drav_ds = load_all_dravidian(dravidian_base, split=split)
            if drav_ds: ds_list.append(drav_ds)
            
        dataset = concatenate_datasets(ds_list)
        num_labels = 2
        class_names = ['Non-Hate', 'Hate']
    else:
        indic_dir = os.path.join(base_dir, 'Datasets', 'Sentiment Training', 'IndicSentiment', 'data')
        ds_list = []
        for split in ['validation', 'test']:
            split_dir = os.path.join(indic_dir, split)
            if os.path.exists(split_dir):
                for fname in os.listdir(split_dir):
                    if fname.endswith('.json'):
                        ds = load_indic_sentiment(os.path.join(split_dir, fname))
                        if ds: ds_list.append(ds)
        dataset = concatenate_datasets(ds_list)
        num_labels = 3
        class_names = ['Positive', 'Negative', 'Neutral']
        
    dataset = dataset.cast_column('label', ClassLabel(num_classes=num_labels))
    
    # Must use identical seed and logic to retrieve the exact test split from train.py
    split = dataset.train_test_split(test_size=0.1, seed=42, stratify_by_column='label')
    eval_ds = split['test']
    
    print("Tokenizing test split...")
    eval_tokenized = tokenize_dataset(eval_ds, tokenizer, max_length=256)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Predict using Trainer
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=data_collator
    )
    
    print("Running inference on test dataset...")
    predictions_output = trainer.predict(eval_tokenized)
    logits = predictions_output.predictions
    y_pred = np.argmax(logits, axis=-1)
    y_true = predictions_output.label_ids
    
    print("\n" + "="*50)
    print(f"CLASSIFICATION REPORT ({args.task.upper()})")
    print("="*50)
    labels_list = list(range(num_labels))
    report = classification_report(y_true, y_pred, labels=labels_list, target_names=class_names, digits=4, zero_division=0)
    print(report)
    
    # Compute Macro F1 manually to exclude classes with 0 support (fixes the 65% deflation issue)
    report_dict = classification_report(y_true, y_pred, labels=labels_list, target_names=class_names, output_dict=True, zero_division=0)
    
    valid_f1s = []
    for cls_name in class_names:
        if report_dict[cls_name]['support'] > 0:
            valid_f1s.append(report_dict[cls_name]['f1-score'])
            
    macro_f1 = sum(valid_f1s) / len(valid_f1s) if valid_f1s else 0.0
    
    # Make sure 'results/plots' directory exists
    plots_dir = os.path.join(base_dir, 'results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_confusion_matrix(y_true, y_pred, class_names, f"Confusion Matrix: {args.task.capitalize()}", os.path.join(plots_dir, f"cm_{args.task}.png"), labels_list)
    plot_comparative_analysis(args.task, macro_f1, os.path.join(plots_dir, f"comparison_{args.task}.png"))
    
    print("\nEvaluation complete! Check the results/plots folder for visual charts.")

if __name__ == "__main__":
    main()

