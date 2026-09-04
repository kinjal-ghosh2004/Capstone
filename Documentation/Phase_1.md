# Phase 1: Dataset Exploration & Standardization

## Overview
Phase 1 focuses on standardizing our labels into a unified format for sentiment and hate speech detection, and performing an Exploratory Data Analysis (EDA) on the datasets to understand their textual properties.

## Tasks Completed
1. **Label Harmonization**
   - Created `src/data/label_mapping.py` which contains generic mapping dictionaries and functions.
   - For **Sentiment**, labels are unified into `0: Positive`, `1: Negative`, `2: Neutral/None`.
   - For **Hate/Offensive**, labels are unified into a binary format:
     - `0 (Non-Offensive)`: Covers labels like `Not_offensive`, `non-antireligion`, `not-kannada`, `not-malayalam`, etc.
     - `1 (Offensive)`: Covers `Offensive_Targeted_Insult_Group`, `antireligion`, `Offensive_Untargetede`, etc.

2. **Exploratory Data Analysis (EDA)**
   - Wrote `src/eda.py` which reads the raw datasets, extracts length statistics (min, max, avg), vocabulary size, and code-mixing artifacts like emojis, mentions (`@`), hashtags (`#`), and URLs.
   - Plotted and saved label distribution visualizations.
   
## EDA Outputs
- Generated JSON stats files for THAR, Dravidian (Kannada, Malayalam, Tamil), and IndicSentiment datasets.
- Generated label distribution bar plots (saved as `.png` images).
- All outputs have been saved to `results/hate_eda/` and `results/sentiment_eda/`.

## Conclusion
With standardized mapping and strong insight into the code-mixed features across our datasets, the data is now prepared for rigorous preprocessing in Phase 2.

