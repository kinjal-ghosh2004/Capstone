import os
# Fix PyTorch CUDA fragmentation before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import sys
import torch
import numpy as np
from datasets import ClassLabel
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback
import torch.nn as nn
# allow imports from parent directory (src)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.loaders import load_thar, load_indic_sentiment
from data.tokenize_utils import get_tokenizer, tokenize_dataset, MODEL_NAME

def compute_metrics(eval_pred):
    """
    Computes accuracy and macro F1 score during evaluation using pure NumPy.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Accuracy
    acc = np.mean(predictions == labels)
    
    # Macro F1
    unique_classes = np.unique(np.concatenate((labels, predictions)))
    f1s = []
    for c in unique_classes:
        tp = np.sum((predictions == c) & (labels == c))
        fp = np.sum((predictions == c) & (labels != c))
        fn = np.sum((predictions != c) & (labels == c))
        
        denominator = 2 * tp + fp + fn
        if denominator == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * tp / denominator)
            
    f1 = np.mean(f1s) if len(f1s) > 0 else 0.0
    
    return {
        'accuracy': acc,
        'f1_macro': f1
    }

class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Check if class_weights is passed and move it to the correct device
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def main():
    parser = argparse.ArgumentParser(description="Fine-tune IndicBERT for Code-Mixed Data")
    parser.add_argument('--task', type=str, required=True, choices=['hate', 'sentiment'], help="Task to train on")
    parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for training/eval")
    parser.add_argument('--lr', type=float, default=2e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    print(f"Loading tokenizer from: {MODEL_NAME}")
    tokenizer = get_tokenizer()
    
    # 1. Load data
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from datasets import concatenate_datasets
    
    if args.task == 'hate':
        print("Loading THAR and Dravidian Hate Speech datasets...")
        ds_list = []
        
        # Load THAR
        thar_path = os.path.join(base_dir, 'Datasets', 'Hate Speech Training', 'THAR', 'THAR-Dataset.csv')
        thar_ds = load_thar(thar_path)
        if thar_ds: ds_list.append(thar_ds)
        
        # Load Dravidian (Kannada, Malayalam, Tamil)
        from data.loaders import load_all_dravidian
        dravidian_base = os.path.join(base_dir, 'Datasets', 'Hate Speech Training', 'Dravidian-Offensive-Language-Identification', 'Datasets')
        for split in ['train', 'dev']:
            drav_ds = load_all_dravidian(dravidian_base, split=split)
            if drav_ds: ds_list.append(drav_ds)
            
        dataset = concatenate_datasets(ds_list)
        num_labels = 2
    else:
        print("Loading ALL IndicSentiment datasets (test and validation splits across 13 languages)...")
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
        
    print(f"Total samples: {len(dataset)}")
    
    # Cast label column to ClassLabel for stratification
    dataset = dataset.cast_column('label', ClassLabel(num_classes=num_labels))
    
    # Train / Eval split
    split = dataset.train_test_split(test_size=0.1, seed=42, stratify_by_column='label')
    train_ds = split['train']
    eval_ds = split['test']
    
    print("Tokenizing data...")
    train_tokenized = tokenize_dataset(train_ds, tokenizer, max_length=128)
    eval_tokenized = tokenize_dataset(eval_ds, tokenizer, max_length=128)
    
    # The tokenize_dataset formats for torch with specific columns, but Trainer handles HF datasets directly.
    # DataCollator automatically pads
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 2. Load Model
    print(f"Loading uninitialized sequence classification model with {num_labels} labels...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels, ignore_mismatched_sizes=True)
    
    output_dir = os.path.join(base_dir, 'results', 'models', args.task)
    
    labels = np.array(train_ds['label'])
    class_weights = []
    total_samples = len(labels)
    n_classes_present = len(np.unique(labels))
    for c in range(num_labels):
        count = np.sum(labels == c)
        weight = total_samples / (n_classes_present * count) if count > 0 else 0.0
        class_weights.append(weight)
    
    # 3. Setup Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to="none", # no wandb for now
        fp16=True,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        # Memory & Performance Optimizations:
        optim="adafactor",           # Extremely memory-efficient optimizer
        dataloader_num_workers=4,    # Parallelize data loading (safe on Windows within __main__)
        gradient_accumulation_steps=16, # Accumulate gradients over 16 steps to simulate batch size of 16
        gradient_checkpointing=True,   # Saves massive memory at the cost of slight compute overhead
    )
    
    # 4. Initialize Trainer
    trainer = CustomTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Evaluating best model...")
    eval_results = trainer.evaluate()
    print(eval_results)
    
    # Save the final best model
    trainer.save_model(os.path.join(output_dir, 'best_model'))
    print(f"Training complete. Model saved to {output_dir}/best_model")

if __name__ == "__main__":
    main()
