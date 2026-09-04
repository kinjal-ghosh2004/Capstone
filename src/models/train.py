import argparse
import os
import sys
import torch
import numpy as np
from datasets import ClassLabel
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback
import torch.nn as nn
# allow imports from parent directory (src)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.loaders import load_thar, load_indic_sentiment
from data.tokenize_utils import get_tokenizer, tokenize_dataset, MODEL_NAME

def compute_metrics(eval_pred):
    """
    Computes accuracy and macro F1 score during evaluation.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    
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
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training/eval")
    parser.add_argument('--lr', type=float, default=2e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    print(f"Loading tokenizer from: {MODEL_NAME}")
    tokenizer = get_tokenizer()
    
    # 1. Load data
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    if args.task == 'hate':
        print("Loading THAR Hate Speech dataset...")
        dataset_path = os.path.join(base_dir, 'Datasets', 'Hate Speech Training', 'THAR', 'THAR-Dataset.csv')
        dataset = load_thar(dataset_path)
        num_labels = 2
    else:
        print("Loading IndicSentiment dataset (Hindi Validation as train for test)...")
        # Just an example path. Ideally, we would load all train splits.
        # But we'll just use what we loaded earlier.
        dataset_path = os.path.join(base_dir, 'Datasets', 'Sentiment Training', 'IndicSentiment', 'data', 'validation', 'hi.json')
        dataset = load_indic_sentiment(dataset_path)
        num_labels = 3
        
    print(f"Total samples: {len(dataset)}")
    
    # Cast label column to ClassLabel for stratification
    dataset = dataset.cast_column('label', ClassLabel(num_classes=num_labels))
    
    # Train / Eval split
    split = dataset.train_test_split(test_size=0.1, seed=42, stratify_by_column='label')
    train_ds = split['train']
    eval_ds = split['test']
    
    print("Tokenizing data...")
    train_tokenized = tokenize_dataset(train_ds, tokenizer, max_length=256)
    eval_tokenized = tokenize_dataset(eval_ds, tokenizer, max_length=256)
    
    # The tokenize_dataset formats for torch with specific columns, but Trainer handles HF datasets directly.
    # DataCollator automatically pads
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 2. Load Model
    print(f"Loading uninitialized sequence classification model with {num_labels} labels...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    
    output_dir = os.path.join(base_dir, 'results', 'models', args.task)
    
    from sklearn.utils.class_weight import compute_class_weight
    labels = train_ds['label']
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights = list(class_weights)
    
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
        weight_decay=0.01
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
