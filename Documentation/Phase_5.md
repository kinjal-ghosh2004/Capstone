# Phase 5: Transfer Learning (Fine-Tuning)

## Overview
Phase 5 implements the fine-tuning loops to adapt our pre-trained `IndicBERT` model to the downstream tasks (Hate Speech and Sentiment Analysis). Using the HuggingFace `Trainer` API, we can efficiently perform gradient updates, evaluate over validation sets, and persist the best performing model.

## Tasks Completed
1. **Training Script (`src/models/train.py`)**
   - Configured an end-to-end training script that accepts CLI arguments for hyperparameters (`--task`, `--epochs`, `--batch_size`, `--lr`).
   - Integrated dynamic data splitting using `datasets.train_test_split`.
   - Setup `DataCollatorWithPadding` to dynamically pad batches during training, heavily optimizing memory footprint compared to static padding.

2. **Trainer Initialization**
   - Mapped the loss and metric computations (Accuracy, Macro F1).
   - Set up `TrainingArguments` to log metrics periodically, evaluate at the end of each epoch, and automatically save the model performing the highest Macro F1 score on the validation split.

3. **Validation & Verification**
   - Ran a dry-run iteration on the CPU to verify loss propagation, tensor formatting, and the absence of out-of-memory or type mismatch errors.

## Next Steps
The codebase is now fully equipped to produce production-grade models! Since deep learning fine-tuning on large datasets (like the 11k+ THAR dataset) requires significant computational resources, the scripts can be executed whenever you have access to a CUDA-enabled GPU (e.g., local NVIDIA card, Google Colab, or Kaggle).

### How to train:
To train the hate speech model:
```bash
python src/models/train.py --task hate --epochs 3 --batch_size 16 --lr 2e-5
```

To train the sentiment model:
```bash
python src/models/train.py --task sentiment --epochs 3 --batch_size 16 --lr 2e-5
```

The fine-tuned weights will be exported to `results/models/hate/best_model` and `results/models/sentiment/best_model` respectively.

This officially completes the fine-tuning architectural phase of the Capstone project!

