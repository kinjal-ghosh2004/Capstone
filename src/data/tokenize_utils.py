import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# IndicBERT tokenizer from local folder
import os
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_NAME = os.path.join(base_dir, 'Datasets', 'Pretraining')

def get_tokenizer(model_name=MODEL_NAME):
    """
    Initializes and returns the tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    return tokenizer

def tokenize_dataset(dataset, tokenizer, max_length=128):
    """
    Tokenizes a HuggingFace Dataset.
    Applies padding and truncation, and returns PyTorch tensors.
    """
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
    
    # Apply tokenizer to the dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Format the dataset to return PyTorch tensors
    columns_to_return = ['input_ids', 'attention_mask', 'label']
    # If using ALBERT (indic-bert), token_type_ids is also commonly returned
    if 'token_type_ids' in tokenized_dataset.features:
        columns_to_return.append('token_type_ids')
        
    tokenized_dataset.set_format(type='torch', columns=columns_to_return)
    return tokenized_dataset

def create_dataloader(tokenized_dataset, batch_size=32, shuffle=True):
    """
    Creates a PyTorch DataLoader from the tokenized dataset.
    """
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
