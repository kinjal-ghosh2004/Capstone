import pandas as pd
import re

def normalize_text(text):
    """
    Applies basic normalization while retaining code-mixing features.
    Retains emojis, hashtags, mentions, slang, repeated characters, and punctuation.
    """
    if not isinstance(text, str):
        return ""
        
    # Lowercase the text
    text = text.lower()
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def preprocess_dataframe(df, text_col):
    """
    Preprocess a pandas DataFrame.
    - Drops missing values in the text column
    - Drops duplicates based on the text column
    - Applies text normalization
    """
    # 1. Drop missing values in the text column
    df = df.dropna(subset=[text_col])
    
    # 2. Filter out empty strings
    df = df[df[text_col].astype(str).str.strip() != '']
    
    # 3. Drop duplicates
    df = df.drop_duplicates(subset=[text_col])
    
    # 4. Normalize text
    df[text_col] = df[text_col].apply(normalize_text)
    
    # Post-normalization filtering of empty texts (just in case)
    df = df[df[text_col] != '']
    
    return df

