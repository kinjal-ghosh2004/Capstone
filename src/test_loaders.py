from data.loaders import load_indic_sentiment, load_thar, load_all_dravidian
from datasets import concatenate_datasets
import os

def test():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print("Testing THAR loader...")
    thar_ds = load_thar(os.path.join(base_dir, 'Datasets', 'Hate Speech Training', 'THAR', 'THAR-Dataset.csv'))
    print(f"THAR Dataset: {len(thar_ds)} samples. Features: {thar_ds.features}")
    
    print("\nTesting Dravidian loader...")
    dravidian_dir = os.path.join(base_dir, 'Datasets', 'Hate Speech Training', 'Dravidian-Offensive-Language-Identification', 'Datasets')
    drav_train = load_all_dravidian(dravidian_dir, split='train')
    if drav_train:
        print(f"Dravidian Train Dataset: {len(drav_train)} samples. Features: {drav_train.features}")
    else:
        print("Dravidian Train Dataset not found.")
        
    print("\nTesting IndicSentiment loader...")
    sent_path = os.path.join(base_dir, 'Datasets', 'Sentiment Training', 'IndicSentiment', 'data', 'validation', 'hi.json')
    sent_ds = load_indic_sentiment(sent_path)
    if sent_ds:
        print(f"IndicSentiment HI Dataset: {len(sent_ds)} samples. Features: {sent_ds.features}")
        
    print("\nSample (THAR):")
    print(thar_ds[0])

if __name__ == '__main__':
    test()

