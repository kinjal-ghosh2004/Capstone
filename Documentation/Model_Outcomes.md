# Model Training Outcomes

This document summarizes the final evaluation metrics for the fine-tuned IndicBERT models on the Code-Mixed Data tasks.

## 1. Hate Speech Detection (THAR Dataset)

The model was fine-tuned for a binary sequence classification task (Hate Speech vs. Non-Hate Speech).

- **Epochs**: 3
- **Learning Rate**: `2e-5`
- **Batch Size**: 16
- **Training Samples**: ~11,549

### Final Evaluation Metrics
- **Accuracy**: 78.01%
- **F1 Score (Macro)**: 0.7798
- **Evaluation Loss**: 0.5035
- **Samples per second**: 127.2

*Note: The model achieved strong performance on the Hate Speech task, with a balanced F1 score tracking closely with overall accuracy.*

---

## 2. Sentiment Analysis (IndicSentiment Hindi)

The model was fine-tuned for a 3-class sequence classification task (Positive, Negative, Neutral). 

- **Epochs**: 3
- **Learning Rate**: `2e-5`
- **Batch Size**: 16
- **Training Samples**: 156 (Validation subset used for training tests)

### Final Evaluation Metrics
- **Accuracy**: 50.00%
- **F1 Score (Macro)**: 0.2319
- **Evaluation Loss**: 1.0888
- **Samples per second**: 109.5

*Note: The evaluation metrics for the sentiment analysis task are currently low due to the extremely small dataset split (only 156 total samples) used during this training run. Training on the full IndicSentiment dataset is recommended to improve the macro F1 score and overall accuracy.*
