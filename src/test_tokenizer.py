from data.loaders import load_thar
from data.tokenize_utils import get_tokenizer, tokenize_dataset, create_dataloader

def test():
    print("Loading THAR dataset...")
    # Load just a subset for quick testing
    thar_ds = load_thar('d:/Capstone/Datasets/Hate Speech Training/THAR/THAR-Dataset.csv')
    thar_ds = thar_ds.select(range(100)) # take only 100 samples
    
    print("Loading tokenizer ai4bharat/indic-bert...")
    tokenizer = get_tokenizer()
    
    print("Tokenizing dataset...")
    tokenized_ds = tokenize_dataset(thar_ds, tokenizer, max_length=64)
    print(f"Features after tokenization: {list(tokenized_ds.features.keys())}")
    
    print("Creating DataLoader...")
    dataloader = create_dataloader(tokenized_ds, batch_size=16)
    
    batch = next(iter(dataloader))
    print("\nBatch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {v.shape}")
        
    print("\nTokenization test successful!")

if __name__ == '__main__':
    test()

