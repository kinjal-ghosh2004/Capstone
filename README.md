# Capstone Project: IndicBERT Fine-Tuning

This repository contains the source code and datasets for fine-tuning IndicBERT on Code-Mixed Data for downstream tasks such as Hate Speech Detection and Sentiment Analysis.

## Training on Google Colab

Google Colab is the recommended environment for training these models as it provides free GPU access.

1. Open a new notebook in [Google Colab](https://colab.research.google.com/) and enable a GPU (`Runtime` > `Change runtime type` > `T4 GPU`).
2. Run the following in the first cell to clone the repository with Git LFS (for large datasets):
   ```bash
   !git lfs install
   !git clone <YOUR-GITHUB-REPO-URL>
   %cd <YOUR-REPO-NAME>
   ```
3. Install the required dependencies:
   ```bash
   !pip install -q transformers datasets accelerate scikit-learn
   ```
4. Run the training script:
   ```bash
   # Train on the Hate Speech task
   !python src/models/train.py --task hate
   
   # Or train on the Sentiment Analysis task
   !python src/models/train.py --task sentiment
   ```

## Repository Structure

- `src/`: Contains the source code for the models, tokenization utils, and data loaders.
- `Datasets/`: Contains the Code-Mixed training and evaluation datasets.
- `results/`: Output directory where trained model checkpoints are saved (ignored by git).

