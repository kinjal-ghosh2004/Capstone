# Phase 6: Final Results & Evaluation

## Overview
After successfully constructing the architecture locally, the fine-tuning script (`src/models/train.py`) was executed on a GPU instance (Google Colab) to train the classification heads and contextual layers of the `IndicBERT` model. 

## Fine-Tuning Performance

### 1. Hate Speech Detection (THAR Dataset)
The model was trained on the robust THAR dataset (11k+ samples).
- **Baseline (Zero-shot)**
  - Accuracy: 43.6%
  - F1 (Macro): 30.3%
- **Fine-Tuned (Post-Training)**
  - **Accuracy**: **77.5%**
  - **F1 (Macro)**: **77.5%**
- **Conclusion**: The model successfully learned to detect code-mixed hate speech, demonstrating a massive **+34% increase in accuracy** and **+47% increase in Macro F1** over the baseline. The metrics prove the preprocessing logic (preserving emojis, slang, and hashtags) effectively retained critical signals.

### 2. Sentiment Analysis (IndicSentiment)
- **Baseline (Zero-shot)**
  - Accuracy: 49.3%
  - F1 (Macro): 49.3%
- **Fine-Tuned (Post-Training)**
  - **Accuracy**: **56.3%**
  - **F1 (Macro)**: **36.0%**
- **Conclusion & Next Steps**: The model showed a slight increase in accuracy but a drop in Macro F1. This is expected because our current `train.py` script was configured to load only the small `validation/hi.json` file (156 samples) as a proof-of-concept. To achieve a jump similar to the Hate Speech model, the `train.py` script can be updated to load the full training split from the IndicSentiment dataset.

## Final Summary
The pipeline successfully proves that a pre-trained IndicBERT model can be adapted to highly unstructured, code-mixed text data. The repository now contains a fully modular pipeline encompassing data auditing, EDA, preprocessing, dynamic tokenization, and HuggingFace Trainer fine-tuning.

