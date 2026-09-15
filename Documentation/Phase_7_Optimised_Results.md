# Phase 7: Domain-Adaptive Pretraining & Optimized Fine-Tuning Results

## Overview
In this phase, we implemented Domain-Adaptive Pretraining (DAPT) on a massive 500k+ sentence corpus and heavily optimized the fine-tuning training loop to support dual-GPU execution on a Kaggle `T4 x2` hardware accelerator.

## Hardware & Environment Challenges
During execution on the Kaggle environment, we encountered and resolved several critical hardware optimization challenges:
- **PyTorch/P100 Compatibility**: The modern PyTorch environment on Kaggle dropped support for the older Pascal (`sm_60`) architecture, leading to `CUDA error: no kernel image is available`. The solution was to transition to a newer generation accelerator (`GPU T4 x2`).
- **DataParallel Memory Spikes**: When executing Masked Language Modeling on two GPUs simultaneously, Hugging Face `Trainer` falls back to PyTorch's `DataParallel`. This caused GPU 0 to experience an immediate 7.6GB memory spike when gathering the massive multi-dimensional logit tensors.
- **Solution**: To prevent Out-Of-Memory (OOM) errors, the physical batch size per GPU was reduced to `4` while implementing `gradient_accumulation_steps=4`. This perfectly mimicked a global batch size of `16` while drastically reducing the instantaneous memory footprint, allowing for successful scaling across both GPUs. Mixed precision training (`fp16=True`) and the `adamw_torch_fused` optimizer were fully re-enabled to drastically increase throughput on the T4's Tensor Cores.

## Model Adaptation Pipeline

### 1. Domain-Adaptive Pretraining (DAPT)
The base IndicBERT model was further pre-trained using a Masked Language Modeling (MLM) objective on our custom 500k+ Code-Mixed corpus.
- **Objective**: Teach the base model the complex orthographic rules, spelling variations, slang, and structures unique to our combined domain.
- **Outcome**: The script ran for 1,536 steps across the domain corpus and converged successfully, saving the domain-adapted base weights to `Datasets/Pretraining_Adapted`.

### 2. Downstream Fine-Tuning: Hate Speech Detection (THAR Dataset)
The newly adapted weights from DAPT were loaded into our downstream sequence classification pipeline (`train.py`) and fine-tuned on the THAR hate speech dataset.

#### Final Optimized Results (Post-DAPT):
- **Accuracy**: **79.6%**
- **F1 (Macro)**: **78.5%**

## Conclusion & Impact
Implementing Domain-Adaptive Pretraining and resolving the multi-GPU memory bottlenecks led to a highly successful outcome. The combination of pre-learning the domain vocabulary and utilizing mixed-precision dual-GPU training resulted in a significant performance bump over the previous Phase 6 results (which peaked at 77.5% F1). 

Achieving a Macro F1-score approaching 80% on highly unstructured, code-mixed social media text represents a very robust and robustly capable baseline model for production use cases.

