import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset as TorchDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.loaders import load_thar, load_indic_sentiment

class SimpleVocab:
    def __init__(self, texts, max_vocab_size=10000):
        self.pad_idx = 0
        self.unk_idx = 1
        self.word2idx = {'<PAD>': self.pad_idx, '<UNK>': self.unk_idx}
        
        counter = Counter()
        for text in texts:
            counter.update(str(text).split())
            
        for word, _ in counter.most_common(max_vocab_size - 2):
            self.word2idx[word] = len(self.word2idx)
            
    def __len__(self):
        return len(self.word2idx)
        
    def encode(self, text, max_len=128):
        tokens = str(text).split()
        idx_seq = [self.word2idx.get(w, self.unk_idx) for w in tokens][:max_len]
        # Pad sequence
        if len(idx_seq) < max_len:
            idx_seq.extend([self.pad_idx] * (max_len - len(idx_seq)))
        return idx_seq

class TextDataset(TorchDataset):
    def __init__(self, texts, labels, vocab, max_len=128):
        self.encoded_texts = [vocab.encode(t, max_len) for t in texts]
        self.labels = labels.tolist()
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return torch.tensor(self.encoded_texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        # Concat the final forward and backward hidden state
        hidden_concat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(hidden_concat)

def train_and_evaluate(model, train_loader, test_loader, num_classes, task_name, labels, plot_path, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 3
    for epoch in range(epochs):
        model.train()
        for texts, targets in train_loader:
            texts, targets = texts.to(device), targets.to(device)
            optimizer.zero_grad()
            out = model(texts)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            
    # Evaluation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for texts, targets in test_loader:
            texts, targets = texts.to(device), targets.to(device)
            out = model(texts)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    print(f"\n--- {task_name} | BiLSTM ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (Macro): {f1:.4f}")
    print(classification_report(all_targets, all_preds, zero_division=0))
    
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f"BiLSTM Confusion Matrix\n({task_name})")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    return acc, prec, rec, f1

def run_bilstm_baseline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    out_dir = os.path.join(base_dir, 'results')
    plot_dir = os.path.join(out_dir, 'plots')
    metrics_log_path = os.path.join(out_dir, 'classical_baseline_metrics.txt')
    
    print("Loading THAR Hate Speech Dataset...")
    thar_path = os.path.join(base_dir, 'Datasets', 'Hate Speech Training', 'THAR', 'THAR-Dataset.csv')
    df_thar = pd.DataFrame(load_thar(thar_path))
    
    print("Loading IndicSentiment Dataset...")
    sent_path = os.path.join(base_dir, 'Datasets', 'Sentiment Training', 'IndicSentiment', 'data', 'validation', 'hi.json')
    df_sent = pd.DataFrame(load_indic_sentiment(sent_path))
    
    tasks = [
        {'name': 'Hate Speech (THAR)', 'df': df_thar, 'labels': ['Non-Hate', 'Hate'], 'num_classes': 2},
        {'name': 'Sentiment (IndicSentiment)', 'df': df_sent, 'labels': ['Positive', 'Negative', 'Neutral'], 'num_classes': 3}
    ]
    
    with open(metrics_log_path, 'a') as f:
        f.write("=== NEURAL BASELINE (BiLSTM) METRICS ===\n\n")
        
    for task in tasks:
        print(f"\nProcessing {task['name']}...")
        df = task['df']
        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
        
        vocab = SimpleVocab(X_train)
        train_ds = TextDataset(X_train, y_train, vocab)
        test_ds = TextDataset(X_test, y_test, vocab)
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
        
        model = BiLSTMClassifier(len(vocab), embed_dim=100, hidden_dim=128, num_classes=task['num_classes']).to(device)
        
        safe_task_name = task['name'].split(' ')[0].lower()
        plot_path = os.path.join(plot_dir, f'cm_bilstm_{safe_task_name}.png')
        
        acc, prec, rec, f1 = train_and_evaluate(model, train_loader, test_loader, task['num_classes'], task['name'], task['labels'], plot_path, device)
        
        with open(metrics_log_path, 'a') as f:
            f.write(f"--- {task['name']} | BiLSTM ---\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Precision (Macro): {prec:.4f}\n")
            f.write(f"Recall (Macro): {rec:.4f}\n")
            f.write(f"F1 (Macro): {f1:.4f}\n\n")

if __name__ == '__main__':
    run_bilstm_baseline()
