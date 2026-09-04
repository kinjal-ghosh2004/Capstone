# Phase 3: Tokenization & Dataloader Optimization

## Overview
Phase 3 focuses on tokenizing the raw code-mixed text using the IndicBERT tokenizer (`ai4bharat/indic-bert`) and preparing PyTorch `DataLoader` objects for the upcoming modeling phases. Since code-mixed text can often have arbitrary lengths, we implemented dynamic padding and truncation.

## Tasks Completed
1. **Tokenization Pipeline (`src/data/tokenize_utils.py`)**
   - Configured `AutoTokenizer` to fetch and initialize the `ai4bharat/indic-bert` subword tokenizer.
   - Built `tokenize_dataset()` which takes a HuggingFace `Dataset` and applies padding to `max_length` (e.g. 128) and truncates text that exceeds it.
   - Converted the HuggingFace datasets into PyTorch tensor formats (`input_ids`, `attention_mask`, `token_type_ids`, `label`).

2. **Dataloader Optimization**
   - Created `create_dataloader()` to wrap the PyTorch-formatted datasets into `torch.utils.data.DataLoader` objects for efficient batching during the model training phase.

3. **Dependency Updates**
   - Discovered that the `ai4bharat/indic-bert` ALBERT model requires `sentencepiece` for the tokenization engine. Installed `sentencepiece` and appended it to `requirements.txt`.

## Important Note regarding Tokenizer
The `ai4bharat/indic-bert` model is hosted as a gated repository on HuggingFace. We bypassed this by utilizing a local offline copy of the model pre-downloaded to `d:/Capstone/Datasets/Pretraining`. The tokenization pipeline is configured to securely load the tokenizer directly from this local repository.

## Conclusion
The data pipeline is fully constructed. We can take raw code-mixed text from the `Datasets/` directory, map labels, preprocess, tokenize, and batch them into PyTorch tensors. We are now ready to begin **Phase 4: Model Baseline (Zero-shot)**.

