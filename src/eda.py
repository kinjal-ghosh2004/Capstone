import os
import pandas as pd
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Add data folder to path for import
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))
from label_mapping import map_sentiment, map_hate

def extract_features(text):
    text = str(text)
    words = text.split()
    length = len(words)
    chars = len(text)
    
    urls = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
    hashtags = len(re.findall(r'#\w+', text))
    mentions = len(re.findall(r'@\w+', text))
    
    # naive emoji detection
    emojis = len(re.findall(r'[^\w\s,\.\?!;:\'"\-\(\)\[\]\{\}\*&^%$#@`~<>\+/=\\|]', text))
    
    return {
        'length': length,
        'chars': chars,
        'urls': urls,
        'hashtags': hashtags,
        'mentions': mentions,
        'emojis': emojis,
        'words': words
    }

def analyze_dataset(df, text_col, label_col, task_type, name):
    if len(df) == 0:
        return
    
    stats = []
    vocab = set()
    mapped_labels = []
    
    for _, row in df.iterrows():
        text = row.get(text_col, '')
        label = row.get(label_col, None)
        
        feat = extract_features(text)
        feat['label'] = label
        
        if task_type == 'sentiment':
            mapped = map_sentiment(label)
        else:
            mapped = map_hate(label)
            
        mapped_labels.append(mapped)
        for w in feat['words']:
            vocab.add(w.lower())
        stats.append(feat)
        
    df_stats = pd.DataFrame(stats)
    df['Mapped_Label'] = mapped_labels
    
    # Save label distribution plot
    plt.figure(figsize=(8, 6))
    sns.countplot(y='Mapped_Label', data=df)
    plt.title(f'Label Distribution for {name}')
    
    out_dir = 'd:/Capstone/results/sentiment_eda' if task_type == 'sentiment' else 'd:/Capstone/results/hate_eda'
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f'{name}_label_dist.png'))
    plt.close()
    
    summary = {
        'Dataset': name,
        'Sample_Count': len(df_stats),
        'Avg_Length': float(df_stats['length'].mean()),
        'Min_Length': int(df_stats['length'].min()),
        'Max_Length': int(df_stats['length'].max()),
        'Vocabulary_Size': len(vocab),
        'Total_URLs': int(df_stats['urls'].sum()),
        'Total_Hashtags': int(df_stats['hashtags'].sum()),
        'Total_Mentions': int(df_stats['mentions'].sum()),
        'Total_Emojis_Approx': int(df_stats['emojis'].sum())
    }
    
    with open(os.path.join(out_dir, f'{name}_stats.json'), 'w') as f:
        json.dump(summary, f, indent=4)
        
    return summary

def main():
    # 1. THAR (Hate)
    df_thar = pd.read_csv('d:/Capstone/Datasets/Hate Speech Training/THAR/THAR-Dataset.csv')
    analyze_dataset(df_thar, 'Comment', 'SubTask1', 'hate', 'THAR')
    
    # 2. Dravidian (Hate) - Kannada
    df_kan = pd.read_csv('d:/Capstone/Datasets/Hate Speech Training/Dravidian-Offensive-Language-Identification/Datasets/Kannada/kannada_offensive_train.csv', sep='\t', header=None, names=['Text', 'Label'])
    analyze_dataset(df_kan, 'Text', 'Label', 'hate', 'Dravidian_Kannada')
    
    # 3. Dravidian (Hate) - Malayalam
    df_mal = pd.read_csv('d:/Capstone/Datasets/Hate Speech Training/Dravidian-Offensive-Language-Identification/Datasets/Malayalam/mal_full_offensive_train.csv', sep='\t', header=None, names=['Text', 'Label'])
    analyze_dataset(df_mal, 'Text', 'Label', 'hate', 'Dravidian_Malayalam')
    
    # 4. Dravidian (Hate) - Tamil
    df_tam = pd.read_csv('d:/Capstone/Datasets/Hate Speech Training/Dravidian-Offensive-Language-Identification/Datasets/Tamil/tamil_offensive_full_train.csv', sep='\t', header=None, names=['Text', 'Label'])
    analyze_dataset(df_tam, 'Text', 'Label', 'hate', 'Dravidian_Tamil')

    # 5. IndicSentiment
    sent_dir = 'd:/Capstone/Datasets/Sentiment Training/IndicSentiment/data/validation'
    sent_data = []
    for f in os.listdir(sent_dir):
        if f.endswith('.json'):
            with open(os.path.join(sent_dir, f), 'r', encoding='utf-8') as fp:
                for line in fp:
                    try:
                        obj = json.loads(line)
                        sent_data.append({'Text': obj.get('INDIC REVIEW', ''), 'Label': obj.get('LABEL', '')})
                    except:
                        pass
    df_sent = pd.DataFrame(sent_data)
    analyze_dataset(df_sent, 'Text', 'Label', 'sentiment', 'IndicSentiment_Validation')

if __name__ == '__main__':
    main()

