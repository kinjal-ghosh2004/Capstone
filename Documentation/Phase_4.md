# Phase 4: Model Baseline (Zero-shot)

## Overview
Phase 4 focuses on evaluating the raw, pre-trained IndicBERT language model on our downstream tasks (Hate Speech and Sentiment Analysis) before applying any task-specific fine-tuning. This establishes the absolute minimum performance floor (baseline) that our subsequent models must beat.

## Tasks Completed
1. **Model Instantiation (`src/models/baseline.py`)**
   - Successfully loaded the pre-trained `IndicBERT` model from the local repository into an `AutoModelForSequenceClassification` wrapper.
   - The HuggingFace library correctly initialized random weights for the classification head since it had never been trained for classification before.

2. **Evaluation Loop**
   - Processed a subset of the THAR dataset (Hate Speech, Binary classification) through the model.
   - Processed the IndicSentiment Hindi dataset (Sentiment, 3-class classification) through the model.
   - Collected the uncalibrated logits, converted them to predictions via `argmax`, and calculated accuracy, macro F1, and full classification reports.

## Baseline Results
- **Hate Speech (THAR)**
  - **Accuracy**: ~43.6%
  - **F1 (Macro)**: ~30.3%
  - *Observation*: Without a trained classification head, the model essentially functions at random chance or slightly worse depending on initialization logic for the binary head.

- **Sentiment (IndicSentiment - Hindi)**
  - **Accuracy**: ~49.3%
  - **F1 (Macro)**: ~49.3%
  - *Observation*: With 3 classes, random chance is roughly 33%, but slight dataset imbalance pushes the raw baseline up slightly.

These metrics have been formally documented in `results/baseline_metrics.txt`.

## Conclusion
We have a concrete, quantitative understanding of the pre-trained model's raw performance. With the baselines established, the next immediate objective is **Phase 5: Transfer Learning (Fine-Tuning)** where we will train the classification heads and contextual layers to push these metrics upward.

