import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# allow imports from parent directory (src)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.loaders import load_thar, load_indic_sentiment

def evaluate_and_plot(model_name, y_true, y_pred, labels, task_name, plot_path):
    print(f"\n--- {task_name} | {model_name} ---")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (Macro): {prec:.4f}")
    print(f"Recall (Macro): {rec:.4f}")
    print(f"F1 (Macro): {f1:.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f"{model_name} Confusion Matrix\n({task_name})")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    return acc, prec, rec, f1

def run_classical_baselines():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    out_dir = os.path.join(base_dir, 'results')
    plot_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    metrics_log_path = os.path.join(out_dir, 'classical_baseline_metrics.txt')
    
    print("Loading THAR Hate Speech Dataset...")
    thar_path = os.path.join(base_dir, 'Datasets', 'Hate Speech Training', 'THAR', 'THAR-Dataset.csv')
    thar_ds = load_thar(thar_path)
    # Convert HF dataset back to pandas for sklearn
    df_thar = pd.DataFrame(thar_ds)
    
    print("Loading IndicSentiment Dataset...")
    sent_path = os.path.join(base_dir, 'Datasets', 'Sentiment Training', 'IndicSentiment', 'data', 'validation', 'hi.json')
    sent_ds = load_indic_sentiment(sent_path)
    df_sent = pd.DataFrame(sent_ds)
    
    tasks = [
        {'name': 'Hate Speech (THAR)', 'df': df_thar, 'labels': ['Non-Hate', 'Hate']},
        {'name': 'Sentiment (IndicSentiment)', 'df': df_sent, 'labels': ['Positive', 'Negative', 'Neutral']}
    ]
    
    with open(metrics_log_path, 'w') as f:
        f.write("=== CLASSICAL BASELINES METRICS ===\n\n")
    
    for task in tasks:
        print(f"\nProcessing {task['name']}...")
        df = task['df']
        
        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
        
        # TF-IDF Vectorizer with char n-grams
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 3), max_features=10000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        models = {
            'TF-IDF + Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'TF-IDF + SVM': SVC(kernel='linear', random_state=42)
        }
        
        for model_name, model in models.items():
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)
            
            safe_task_name = task['name'].split(' ')[0].lower()
            safe_model_name = "lr" if "Logistic" in model_name else "svm"
            plot_path = os.path.join(plot_dir, f'cm_tfidf_{safe_model_name}_{safe_task_name}.png')
            
            acc, prec, rec, f1 = evaluate_and_plot(model_name, y_test, y_pred, task['labels'], task['name'], plot_path)
            
            with open(metrics_log_path, 'a') as f:
                f.write(f"--- {task['name']} | {model_name} ---\n")
                f.write(f"Accuracy: {acc:.4f}\n")
                f.write(f"Precision (Macro): {prec:.4f}\n")
                f.write(f"Recall (Macro): {rec:.4f}\n")
                f.write(f"F1 (Macro): {f1:.4f}\n\n")

if __name__ == '__main__':
    run_classical_baselines()
