# IndicBERT Domain-Adaptive Pretraining & Fine-Tuning on Code-Mixed Data

This repository contains an advanced pipeline for training **IndicBERT** on highly unstructured, code-mixed social media text. It features a robust two-stage training process designed to adapt the model to domain-specific slang, emojis, and spelling variations, leading to state-of-the-art results on downstream tasks like **Hate Speech Detection** and **Sentiment Analysis**.

## 🚀 Key Results (Optimized Pipeline)
After applying Domain-Adaptive Pretraining (DAPT) and scaling across dual T4 GPUs, the model achieved the following metrics on the test splits:

- **Hate Speech Detection (THAR Dataset)**
  - Accuracy: **79.6%**
  - F1 (Macro): **78.5%**
- **Sentiment Analysis (IndicSentiment)**
  - Accuracy: **98.6%**
  - F1 (Macro): **98.6%**

## 🏗️ Architecture & Pipeline
1. **Domain-Adaptive Pretraining (DAPT)**: The base IndicBERT model is pre-trained using Masked Language Modeling (MLM) on a massive 500k+ sentence corpus of pure code-mixed text to learn the target domain.
2. **Downstream Fine-Tuning**: The adapted weights are loaded and fine-tuned on task-specific sequence classification datasets.
3. **Hardware Optimizations**: The scripts are strictly optimized for dual-GPU execution (e.g., Kaggle `T4 x2`), utilizing Mixed Precision (`fp16`), Fused Adam, and precise Gradient Accumulation to prevent `DataParallel` OOM errors.

## 💻 Running on Kaggle / Google Colab

This pipeline is recommended to be run on **Kaggle** (using `GPU T4 x2`) or Google Colab (using `T4`).

### 1. Setup Environment
Clone the repository and install the dependencies:
```bash
!git clone <YOUR-GITHUB-REPO-URL>
%cd <YOUR-REPO-NAME>
!pip install -q transformers datasets accelerate scikit-learn
```

### 2. Stage 1: Domain-Adaptive Pretraining (DAPT)
Run the pretraining script first to adapt the base IndicBERT model to the custom code-mixed corpus. 
*Note: Make sure your `Datasets/corpus.txt` is populated before running.*
```bash
!python src/models/domain_adaptation.py
```
*(This will save the new adapted base model to `Datasets/Pretraining_Adapted`)*

### 3. Stage 2: Fine-Tuning
Once DAPT is complete, ensure `tokenize_utils.py` is pointed to the `Pretraining_Adapted` model, then execute the fine-tuning script on your desired task:

```bash
# Train the Hate Speech classifier
!python src/models/train.py --task hate

# Train the Sentiment Analysis classifier
!python src/models/train.py --task sentiment
```

## 📂 Repository Structure

- `src/models/domain_adaptation.py`: Pipeline for Masked Language Modeling (DAPT).
- `src/models/train.py`: Pipeline for fine-tuning Sequence Classification tasks.
- `src/data/tokenize_utils.py`: Centralized tokenization & dynamic truncation logic.
- `Datasets/`: Directory housing the THAR, IndicSentiment, and raw corpora datasets.
- `Documentation/`: Detailed phase-by-phase markdown records of the model's evolution, benchmarks, and architecture decisions.
- `results/`: Ignored directory where final `.bin` checkpoints and best models are saved to prevent repository bloat.
