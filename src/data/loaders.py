import pandas as pd
import json
import os
from datasets import Dataset, concatenate_datasets

from .label_mapping import map_sentiment, map_hate
from .preprocessing import preprocess_dataframe

def load_indic_sentiment(file_path):
    """
    Load an IndicSentiment JSONL file.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                text = obj.get('INDIC REVIEW', '')
                label = obj.get('LABEL', '')
                mapped_label = map_sentiment(label)
                if mapped_label is not None:
                    data.append({'text': text, 'label': mapped_label})
            except:
                continue
    
    df = pd.DataFrame(data)
    if not df.empty:
        df = preprocess_dataframe(df, 'text')
    
    return Dataset.from_pandas(df, preserve_index=False) if not df.empty else None

def load_thar(file_path):
    """
    Load THAR dataset.
    """
    df = pd.read_csv(file_path)
    df['label'] = df['SubTask1'].apply(map_hate)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    df = df.rename(columns={'Comment': 'text'})
    df = preprocess_dataframe(df, 'text')
    
    return Dataset.from_pandas(df[['text', 'label']], preserve_index=False) if not df.empty else None

def load_dravidian(file_path):
    """
    Load Dravidian Kannada, Malayalam, or Tamil CSV/TSV files.
    """
    # They are tab separated with no header
    df = pd.read_csv(file_path, sep='\t', header=None, names=['text', 'original_label'])
    df['label'] = df['original_label'].apply(map_hate)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    df = preprocess_dataframe(df, 'text')
    return Dataset.from_pandas(df[['text', 'label']], preserve_index=False) if not df.empty else None

def load_all_dravidian(base_dir, split='train'):
    """
    Loads Kannada, Malayalam, and Tamil for a specific split and combines them.
    split can be 'train', 'dev', 'test'
    """
    ds_list = []
    
    # Path construction according to dataset layout
    kan = os.path.join(base_dir, 'Kannada', f'kannada_offensive_{split}.csv')
    mal = os.path.join(base_dir, 'Malayalam', f'mal_full_offensive_{split}.csv')
    tam = os.path.join(base_dir, 'Tamil', f'tamil_offensive_full_{split}.csv')
    
    for path in [kan, mal, tam]:
        if os.path.exists(path):
            ds = load_dravidian(path)
            if ds:
                ds_list.append(ds)
                
    if ds_list:
        return concatenate_datasets(ds_list)
    return None

