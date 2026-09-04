import os
import sys
import argparse
import torch
from transformers import (
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

# allow imports from parent directory (src)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.tokenize_utils import get_tokenizer, MODEL_NAME

def main():
    parser = argparse.ArgumentParser(description="Domain-Adaptive Pretraining (DAPT) for IndicBERT")
    parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    corpus_path = os.path.join(base_dir, 'Datasets', 'corpus.txt')
    
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found at {corpus_path}. Please ensure all text data is combined into this file.")
        
    print(f"Loading tokenizer from: {MODEL_NAME}")
    tokenizer = get_tokenizer()

    print(f"Loading raw text dataset from {corpus_path}...")
    # Load dataset using HF Datasets
    raw_datasets = load_dataset('text', data_files={'train': corpus_path})
    
    # We will take 5% of the data for evaluation of the MLM task
    split = raw_datasets['train'].train_test_split(test_size=0.05, seed=42)
    
    print("Tokenizing data...")
    def tokenize_function(examples):
        # We don't pad here, DataCollatorForLanguageModeling will handle padding dynamically
        return tokenizer(examples['text'], truncation=True, max_length=256)
        
    tokenized_datasets = split.map(tokenize_function, batched=True, remove_columns=["text"])
    
    print("Loading AutoModelForMaskedLM...")
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    
    # Data collator for Masked Language Modeling (15% masking probability)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    output_dir = os.path.join(base_dir, 'results', 'models', 'dapt')
    save_model_dir = os.path.join(base_dir, 'Datasets', 'Pretraining_Adapted')
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        fp16=True, # Mixed precision for faster training
        warmup_steps=500,
        weight_decay=0.01,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
    )
    
    print("Starting Domain-Adaptive Pretraining...")
    trainer.train()
    
    print("Evaluating Masked Language Modeling performance...")
    eval_results = trainer.evaluate()
    import math
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    
    print(f"Saving adapted base model to {save_model_dir}...")
    trainer.save_model(save_model_dir)
    tokenizer.save_pretrained(save_model_dir)
    
    print("Done! You can now use this adapted model for fine-tuning by changing MODEL_NAME in tokenize_utils.py.")

if __name__ == "__main__":
    main()

