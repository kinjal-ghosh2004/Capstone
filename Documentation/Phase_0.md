# Phase 0: Project Setup & Dataset Audit

## Overview
Phase 0 focuses on establishing the project environment and conducting a high-level audit of the existing datasets to understand file formats, missing values, length constraints, and class distributions.

## Tasks Completed
1. **Environment Setup**
   - Initialized a Python virtual environment (`.venv`).
   - Installed required libraries via `requirements.txt` (PyTorch, Transformers, Datasets, Pandas, Scikit-learn, Matplotlib, Seaborn, NLTK, WandB, Gradio).
   - Created the project directory structure (`notebooks`, `src`, `models`, `results`, `experiments`, `app`).

2. **Dataset Audit**
   - Executed an automated dataset audit script (`src/audit.py`) on 47 dataset files across the `Datasets/` directory.
   - The script successfully identified `.csv`, `.tsv`, and `.json` formats and logged the required statistics.
   - Output saved to `results/dataset_audit.csv`.

## Datasets Audited
- **Dravidian-Offensive-Language-Identification**: Comprises Kannada, Malayalam, and Tamil train/test/dev sets, alongside the OLID English offensive detection sets.
- **Hinglish-Offensive-Text-Classification**: Contains a profanity list to be potentially utilized for heuristics.
- **THAR (Hate Speech)**: Contains 11.5k code-mixed Hindi/English samples targeted at various religious lines.
- **IndicSentiment**: Includes multi-lingual Indian sentiment (Positive/Negative/None) datasets structured in JSON-lines format.

## Conclusion
The project environment is successfully set up and we have a comprehensive understanding of the dataset files, sizes, and formats.

