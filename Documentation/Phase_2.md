# Phase 2: Data Cleaning & Preprocessing

## Overview
Phase 2 focused on creating a resilient and robust data preprocessing pipeline. Given the nature of code-mixed languages and hate-speech/sentiment data, standard text-cleaning approaches (which often strip special characters) are insufficient and risk destroying valuable context. 

## Tasks Completed
1. **Preprocessing Pipeline (`src/data/preprocessing.py`)**
   - Implemented `normalize_text` to lowercase text and standardize spacing.
   - Crucially **preserved code-mixed artifacts**: emojis, hashtags, mentions, slang, informal spellings, and repeated characters, all of which are essential signals for our downstream tasks.
   - Designed `preprocess_dataframe` to seamlessly handle missing values, drop duplicate entries, and filter empty strings.

2. **Dataset Loaders (`src/data/loaders.py`)**
   - Built custom loading functions for each dataset structure:
     - `load_indic_sentiment()`: Processes JSONL files.
     - `load_thar()`: Processes CSV files with targeted hate speech mapping.
     - `load_dravidian()` / `load_all_dravidian()`: Processes TSV/CSV formats combining train/dev/test splits for Kannada, Malayalam, and Tamil.
   - Intertwined the data loaders with `label_mapping.py` to automatically yield the harmonized integer classes (e.g., `0`, `1`, `2`).
   - The loaders return HuggingFace `Dataset` objects ready for the tokenization process.

3. **Integration Testing**
   - Created `src/test_loaders.py` to validate the extraction.
   - Successfully loaded the THAR dataset (11,549 samples), combined Dravidian subsets, and sample IndicSentiment files, confirming the stability of the unified output formats.

## Conclusion
The data is now clean, consistent, normalized, and encapsulated within HuggingFace `Dataset` wrappers while retaining the raw contextual power of code-mixing.

