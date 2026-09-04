# Capstone Project Plan
## Sentiment Analysis and Hate Speech Detection in Code-Mixed Indian Languages

> **Project duration:** Approximately 10 months / 52 weeks  
> **Primary base model:** IndicBERT  
> **Primary tasks:** Sentiment Analysis + Hate/Offensive Language Detection

---

## 1. Project Overview

The project aims to develop an NLP system capable of analyzing **code-mixed Indian-language text** for:

1. **Sentiment** — e.g. Positive, Negative, Neutral
2. **Hate/Offensive Language** — based on the label definitions supported by the selected datasets

The project will use **IndicBERT as the primary pretrained language model**, followed by task-specific fine-tuning and, if feasible, continued masked-language-model adaptation to code-mixed text.

The development strategy is intentionally incremental:

```text
IndicBERT
    ↓
Dataset Audit
    ↓
Preprocessing & Label Harmonization
    ↓
Classical / Neural Baselines
    ↓
IndicBERT Sentiment Model
    ↓
IndicBERT Hate/Offensive Model
    ↓
Cross-Dataset Evaluation
    ↓
Code-Mixed Adaptation
    ↓
Multi-Task Learning
    ↓
Explainability + Error Analysis
    ↓
Final Model
    ↓
Web Application
    ↓
Final Evaluation + Dissertation
```

---

# 2. Dataset Structure

The acquired datasets are organized under the `Datasets` directory.

Recommended final structure:

```text
Capstone/
│
├── Datasets/
│   ├── Base Model/
│   │   └── IndicBERT/
│   │
│   ├── Sentiment Training/
│   │   ├── GLUECoS/
│   │   └── IndicSentiment/
│   │
│   └── Hate Speech Training/
│       ├── Dravidian-Offensive-Language-Identification/
│       ├── Hinglish-Offensive-Text-Classification/
│       └── THAR/
│
├── notebooks/
├── src/
├── models/
├── results/
├── experiments/
├── app/
└── README.md
```

### Important terminology

IndicBERT is a **pretrained model**, not normally a pretraining dataset.

Therefore, the `Pretraining/` directory should only be used if a separate **unlabeled code-mixed corpus** is later used for continued masked-language-model/domain-adaptive pretraining.

If no such corpus is used, `Base Model/IndicBERT/` is the more accurate structure.

---

# 3. Phase 0 — Project Setup & Dataset Audit

**Duration:** Week 1–2

## Objectives

- Set up the Python/ML environment.
- Verify every acquired dataset.
- Understand dataset formats, labels, languages and splits.
- Identify incompatibilities between datasets before training.

## Tasks

### Environment

Recommended stack:

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK

Optional:

- Weights & Biases
- Gradio / Streamlit

### Dataset audit

For every dataset record:

- File format
- Number of samples
- Column names
- Text column
- Language(s)
- Script
- Label column
- Label definitions
- Class distribution
- Existing train/validation/test splits
- Missing values
- Duplicate samples
- Average text length
- Maximum text length

Create:

```text
results/
└── dataset_audit.csv
```

## Deliverable

A complete dataset audit report describing every dataset and its label scheme.

---

# 4. Phase 1 — Dataset Exploration & Standardization

**Duration:** Week 3–5

## Objectives

Understand the linguistic and statistical characteristics of the data and create a consistent representation across datasets.

## Tasks

### 4.1 Exploratory Data Analysis

For each dataset calculate:

- Sample count
- Average sentence length
- Minimum/maximum length
- Vocabulary size
- Token/word statistics
- Language distribution where available
- English/Indian-language mixing
- Emoji frequency
- URL frequency
- Hashtag frequency
- Mention frequency
- Punctuation patterns
- Repeated-character patterns
- Common words/slang

### 4.2 Label distribution

Visualize the class distribution for every dataset.

For sentiment, possible categories may include:

```text
Positive
Negative
Neutral
```

For offensive/hate datasets, categories will depend on the actual dataset definitions.

**Do not assume that `offensive`, `hate`, `abusive`, and `non-offensive` are interchangeable.**

### 4.3 Label harmonization

Create a documented mapping for labels that genuinely represent the same task.

Example:

```python
sentiment_mapping = {
    "positive": 0,
    "negative": 1,
    "neutral": 2
}
```

Example for a binary task:

```python
hate_mapping = {
    "non-offensive": 0,
    "offensive": 1
}
```

The exact mappings must be based on the original dataset documentation.

## Deliverables

```text
results/
├── dataset_audit.csv
├── sentiment_eda/
└── hate_eda/
```

---

# 5. Phase 2 — Data Cleaning & Preprocessing

**Duration:** Week 6–8

## Objectives

Build a reusable preprocessing pipeline without destroying the linguistic information that makes code-mixed text difficult.

## Pipeline

```text
Raw Dataset
    ↓
Missing-value handling
    ↓
Corrupted sample removal
    ↓
Duplicate detection
    ↓
Text normalization
    ↓
Code-mixing preservation
    ↓
Label normalization
    ↓
Dataset splitting
    ↓
Tokenization
```

## Important preprocessing principle

Do **not** aggressively clean code-mixed text.

For example:

```text
"Ye movie bahut mast hai 😂"
```

should not automatically become:

```text
"This movie is very good"
```

The model needs to learn code-switching, transliteration, slang, emojis and spelling variations.

Be careful when modifying:

- Emojis
- Hashtags
- Mentions
- Slang
- Repeated characters
- Transliteration
- Informal spellings
- Code-switched words

## Deliverables

Create reusable preprocessing modules:

```text
src/
└── data/
    ├── loaders.py
    ├── preprocessing.py
    └── label_mapping.py
```

---

# 6. Phase 3 — Classical & Neural Baselines

**Duration:** Week 9–10

Before using IndicBERT, establish baseline performance.

## Baseline 1

```text
TF-IDF
   ↓
Logistic Regression
   ↓
Classification
```

## Baseline 2

```text
TF-IDF
   ↓
SVM
   ↓
Classification
```

## Optional neural baseline

```text
Embedding
   ↓
BiLSTM
   ↓
Classification
```

Run these baselines for both:

- Sentiment
- Hate/offensive language

## Metrics

Record:

- Accuracy
- Precision
- Recall
- F1-score
- Macro-F1
- Weighted-F1
- Confusion matrix

## Deliverable

A baseline results table that later becomes part of the final research comparison.

---

# 7. Phase 4 — IndicBERT Sentiment Baseline

**Duration:** Week 11–14

## Objective

Fine-tune IndicBERT for sentiment analysis.

### Experiment S1 — GLUECoS

```text
GLUECoS
   ↓
Preprocessing
   ↓
IndicBERT
   ↓
Classification Head
   ↓
Sentiment
```

Record all evaluation metrics.

### Experiment S2 — IndicSentiment

```text
IndicSentiment
   ↓
Preprocessing
   ↓
IndicBERT
   ↓
Classification Head
   ↓
Sentiment
```

### Experiment S3 — Combined sentiment data

If the label definitions and task formulation are compatible:

```text
GLUECoS
    +
IndicSentiment
    ↓
Unified Dataset
    ↓
IndicBERT
    ↓
Sentiment Classifier
```

Compare:

```text
GLUECoS only
IndicSentiment only
GLUECoS + IndicSentiment
```

## Deliverable

A trained sentiment model and a complete experiment log.

---

# 8. Phase 5 — IndicBERT Hate/Offensive Language Model

**Duration:** Week 15–18

## Objective

Develop the second major task-specific model.

### Experiment H1

```text
IndicBERT
    ↓
Dravidian-Offensive-Language-Identification
    ↓
Classifier
```

### Experiment H2

```text
IndicBERT
    ↓
Hinglish-Offensive-Text-Classification
    ↓
Classifier
```

### Experiment H3

```text
IndicBERT
    ↓
THAR
    ↓
Classifier
```

### Experiment H4 — Combined training

Only combine datasets after carefully checking whether their labels have compatible semantic definitions.

```text
Dataset 1
    +
Dataset 2
    +
Dataset 3
    ↓
Unified Hate/Offensive Dataset
    ↓
IndicBERT
    ↓
Hate/Offensive Classifier
```

## Deliverable

A trained hate/offensive-language model plus individual and combined experiment results.

---

# 9. Phase 6 — Cross-Dataset & Cross-Language Evaluation

**Duration:** Week 19–22

This phase is especially important because it tests whether the model actually generalizes instead of simply memorizing dataset-specific patterns.

## Example experiments

### Sentiment

```text
Train: GLUECoS
Test: IndicSentiment
```

```text
Train: IndicSentiment
Test: GLUECoS
```

### Hate/offensive language

```text
Train: Hinglish
Test: THAR
```

```text
Train: Dravidian dataset
Test: THAR
```

The exact experiments depend on compatible label definitions and language coverage.

## Research question

> Can an IndicBERT-based model trained on one code-mixed Indian-language dataset generalize to another dataset with different language, domain or annotation characteristics?

## Deliverable

Cross-dataset generalization matrix.

---

# 10. Phase 7 — Code-Mixed Domain Adaptation

**Duration:** Week 23–26

## Objective

Investigate whether adapting IndicBERT to code-mixed Indian-language text improves downstream performance.

If a suitable **unlabeled code-mixed corpus** is available:

```text
IndicBERT
    ↓
Continued Masked Language Model Training
    ↓
Code-Mixed Adapted IndicBERT
    ↓
Fine-tuning
    ├── Sentiment
    └── Hate/Offensive
```

Compare:

```text
Original IndicBERT
        vs.
Code-Mixed Adapted IndicBERT
```

## Research question

> Does continued language/domain adaptation on code-mixed Indian-language text improve sentiment and offensive-language detection?

## Important

Do not call this "pretraining IndicBERT" unless the model is actually being pretrained from the beginning. A better term is:

**continued pretraining**, **domain-adaptive pretraining**, or **task/domain adaptation**.

---

# 11. Phase 8 — Multi-Task Learning

**Duration:** Week 27–31

## Objective

Investigate whether sentiment and hate/offensive-language detection can benefit from a shared language representation.

Proposed architecture:

```text
                 ┌───────────────────┐
                 │     IndicBERT     │
                 │  Shared Encoder   │
                 └─────────┬─────────┘
                           │
                 Shared representation
                      /           \
                     /             \
                    ↓               ↓
          Sentiment Head      Hate Head
                    ↓               ↓
             Sentiment        Hate/Offensive
```

The model has:

- One shared IndicBERT encoder
- One sentiment classification head
- One hate/offensive classification head

## Compare

```text
Independent Sentiment Model
            vs.
Multi-Task Sentiment Model
```

and:

```text
Independent Hate Model
            vs.
Multi-Task Hate Model
```

## Research question

> Does joint learning of sentiment and hate/offensive-language detection improve performance compared with independently trained models?

## Deliverable

A working multi-task architecture and comparison with the individual models.

---

# 12. Phase 9 — Model Comparison

**Duration:** Week 32–34

Build a consolidated comparison table.

| Model | Sentiment Macro-F1 | Hate Macro-F1 | Notes |
|---|---:|---:|---|
| TF-IDF + Logistic Regression | TBD | TBD | Classical baseline |
| TF-IDF + SVM | TBD | TBD | Classical baseline |
| BiLSTM | TBD | TBD | Neural baseline |
| mBERT | TBD | TBD | Optional |
| XLM-R | TBD | TBD | Optional |
| IndicBERT | TBD | TBD | Primary model |
| Adapted IndicBERT | TBD | TBD | If adaptation is performed |
| Multi-Task IndicBERT | TBD | TBD | Proposed architecture |

Do not force every optional model into the project if compute resources are limited.

The most important comparison is:

```text
Classical baseline
       ↓
IndicBERT
       ↓
Adapted IndicBERT
       ↓
Multi-Task IndicBERT
```

---

# 13. Phase 10 — Error Analysis

**Duration:** Week 35–37

Metrics alone are not enough.

Analyze incorrect predictions and categorize the errors.

## Error categories

### Sarcasm

```text
"Wow kya amazing service hai 🙄"
```

### Slang

```text
"bhai kya bakchodi hai"
```

### Spelling variation

```text
"bohottttt acchaaaa"
```

### Code-switching

```text
"dei this is seriously irritating"
```

### Context dependency

```text
"nice one bro"
```

The same phrase may be positive or sarcastic depending on context.

### Implicit offensive/hateful meaning

A statement may target a person or group without containing obvious offensive keywords.

### Annotation ambiguity

Some examples may be genuinely difficult or inconsistently labeled.

## Deliverable

A categorized error-analysis report containing representative examples and explanations.

---

# 14. Phase 11 — Explainability

**Duration:** Week 38–40

## Objective

Show why the model made a prediction instead of treating it as a black box.

Potential techniques:

- Attention visualization
- SHAP
- Integrated Gradients
- Token importance

Example:

```text
Input:
"nee romba worst da"

Prediction:
Offensive

Important token:
"worst"
```

Example:

```text
Input:
"movie semma mass 🔥"

Prediction:
Positive

Potentially important tokens:
"semma"
"mass"
"🔥"
```

The final explainability method should be selected based on compatibility with the architecture and available compute.

## Deliverable

Explainability examples integrated into the evaluation/report.

---

# 15. Phase 12 — Final Unified System

**Duration:** Week 41–43

Build the final inference pipeline.

```text
                 User Input
                     ↓
               Preprocessing
                     ↓
                  IndicBERT
                     ↓
             Shared Representation
                /           \
               /             \
              ↓               ↓
      Sentiment Head      Hate Head
              ↓               ↓
        Sentiment       Hate/Offensive
```

Example output:

```text
Input:
"Bro this movie semma waste 😂"

Sentiment:
Negative

Confidence:
94.2%

Hate/Offensive:
Non-Offensive

Confidence:
91.7%
```

The exact output labels and confidence values will come from the trained model.

---

# 16. Phase 13 — Application Development

**Duration:** Week 44–46

Create a simple user interface using **Gradio** or **Streamlit**.

## Basic interface

```text
╔════════════════════════════════════╗
║ Indian Code-Mixed Text Analyzer   ║
╠════════════════════════════════════╣
║                                    ║
║ Enter text:                        ║
║ ┌────────────────────────────────┐ ║
║ │ Bro this movie semma waste 😂  │ ║
║ └────────────────────────────────┘ ║
║                                    ║
║          [ Analyze ]                ║
║                                    ║
║ Sentiment: Negative                ║
║ Confidence: XX.X%                  ║
║                                    ║
║ Offensive: No                      ║
║ Confidence: XX.X%                  ║
╚════════════════════════════════════╝
```

## Optional features

- Confidence visualization
- Detected language/script
- Token explanation
- Example inputs
- Prediction history
- Error/report feedback

Keep the application simple. The **ML research should remain the centerpiece**.

---

# 17. Phase 14 — Final Evaluation

**Duration:** Week 47–48

Evaluate the final model on held-out test data.

## Sentiment metrics

- Accuracy
- Precision
- Recall
- Macro-F1
- Weighted-F1
- Confusion matrix

## Hate/offensive metrics

- Accuracy
- Precision
- Recall
- Macro-F1
- Weighted-F1
- Confusion matrix

## Robustness tests

Where appropriate, evaluate sensitivity to:

- Spelling variations
- Emojis
- Slang
- Code-switching
- Transliteration
- Noisy text
- Repeated characters

---

# 18. Phase 15 — Dissertation / Report

**Duration:** Week 49–51

Recommended structure:

```text
1. Introduction

2. Problem Statement

3. Objectives

4. Literature Review

5. Dataset Description

6. Data Preprocessing

7. Proposed Methodology

8. Baseline Models

9. IndicBERT Fine-Tuning

10. Domain/Code-Mixed Adaptation

11. Multi-Task Learning

12. Experimental Setup

13. Results

14. Error Analysis

15. Explainability

16. Application / Deployment

17. Limitations

18. Future Work

19. Conclusion

20. References
```

---

# 19. Phase 16 — Final Presentation & Demo

**Duration:** Week 52

The final presentation should tell a clear story:

```text
Problem
   ↓
Why code-mixed Indian languages?
   ↓
Limitations of existing approaches
   ↓
Datasets
   ↓
IndicBERT
   ↓
Baseline Models
   ↓
Task-Specific Fine-Tuning
   ↓
Cross-Dataset Experiments
   ↓
Code-Mixed Adaptation
   ↓
Multi-Task Learning
   ↓
Results
   ↓
Error Analysis
   ↓
Explainability
   ↓
Application
   ↓
Conclusion
```

## Demo strategy

Use representative code-mixed examples rather than only clean English sentences.

Examples:

```text
"dei this movie was semma mass 🔥"
```

```text
"bro worst movie ever 😂"
```

```text
"nee romba loosu da"
```

Show:

- Input
- Sentiment prediction
- Hate/offensive prediction
- Confidence
- Optional token explanation

---

# 20. Final Project Architecture

Recommended software structure:

```text
Capstone/
│
├── Datasets/
│   ├── Base Model/
│   │   └── IndicBERT/
│   │
│   ├── Sentiment Training/
│   │   ├── GLUECoS/
│   │   └── IndicSentiment/
│   │
│   └── Hate Speech Training/
│       ├── Dravidian-Offensive-Language-Identification/
│       ├── Hinglish-Offensive-Text-Classification/
│       └── THAR/
│
├── src/
│   ├── data/
│   │   ├── loaders.py
│   │   ├── preprocessing.py
│   │   └── label_mapping.py
│   │
│   ├── models/
│   │   ├── baseline.py
│   │   ├── indicbert.py
│   │   └── multitask.py
│   │
│   ├── training/
│   │   ├── train_sentiment.py
│   │   ├── train_hate.py
│   │   └── train_multitask.py
│   │
│   └── evaluation/
│       ├── metrics.py
│       ├── confusion_matrix.py
│       └── error_analysis.py
│
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_sentiment_eda.ipynb
│   ├── 03_hate_eda.ipynb
│   ├── 04_sentiment_baseline.ipynb
│   └── 05_hate_baseline.ipynb
│
├── experiments/
│   ├── sentiment/
│   ├── hate/
│   └── multitask/
│
├── models/
│
├── results/
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

# 21. Ten-Month Timeline

| Month | Weeks | Phase | Main Deliverable |
|---|---:|---|---|
| Month 1 | 1–5 | Setup, audit, EDA | Dataset report |
| Month 2 | 6–8 | Cleaning & preprocessing | Unified preprocessing pipeline |
| Month 3 | 9–10 | Baselines | Baseline results |
| Month 3–4 | 11–14 | Sentiment training | Sentiment model |
| Month 4–5 | 15–18 | Hate training | Hate/offensive model |
| Month 5–6 | 19–22 | Cross-dataset evaluation | Generalization results |
| Month 6 | 23–26 | Code-mixed adaptation | Adapted IndicBERT |
| Month 7 | 27–31 | Multi-task learning | Proposed model |
| Month 8 | 32–34 | Model comparison | Final comparison |
| Month 8–9 | 35–40 | Error analysis + explainability | Analysis report |
| Month 9 | 41–46 | Unified system + application | Working application |
| Month 10 | 47–51 | Final evaluation + dissertation | Final report |
| Month 10 | 52 | Presentation/demo | Final capstone presentation |

---

# 22. Core Research Questions

The project should be driven by measurable research questions rather than simply "build an NLP model."

### RQ1 — Base model performance

> How effectively can IndicBERT detect sentiment and hate/offensive language in code-mixed Indian-language text?

### RQ2 — Dataset combination

> Does combining compatible datasets improve performance compared with training on individual datasets?

### RQ3 — Generalization

> How well does a model trained on one code-mixed dataset generalize to another dataset or language/domain?

### RQ4 — Domain adaptation

> Does continued adaptation of IndicBERT to code-mixed Indian-language text improve downstream classification performance?

### RQ5 — Multi-task learning

> Can jointly learning sentiment and hate/offensive-language detection improve performance compared with independently trained models?

### RQ6 — Robustness

> How does the final system perform when code-mixed text contains slang, spelling variations, emojis, transliteration and other forms of linguistic noise?

---

# 23. Key Experimental Comparisons

The project should ultimately establish a progression such as:

```text
Classical Baseline
       ↓
IndicBERT
       ↓
IndicBERT + Dataset Combination
       ↓
Adapted IndicBERT
       ↓
Multi-Task IndicBERT
```

The goal is **not** to assume that every additional technique will improve the score.

If a simpler model performs better, that is a valid research result.

For example:

```text
IndicBERT          → 91.2 Macro-F1
Adapted IndicBERT  → 92.0 Macro-F1
Multi-Task         → 91.6 Macro-F1
```

The conclusion would not be "multi-task learning failed."

Instead, investigate **why** joint learning did not outperform the specialized model.

That analysis is part of the research contribution.

---

# 24. Final Deliverables Checklist

## Dataset

- [ ] Dataset audit
- [ ] Dataset statistics
- [ ] Label mapping
- [ ] Data cleaning pipeline
- [ ] Train/validation/test strategy

## Models

- [ ] TF-IDF + Logistic Regression
- [ ] TF-IDF + SVM
- [ ] Optional BiLSTM
- [ ] IndicBERT sentiment model
- [ ] IndicBERT hate/offensive model
- [ ] Optional adapted IndicBERT
- [ ] Multi-task IndicBERT

## Experiments

- [ ] Individual dataset training
- [ ] Combined dataset training
- [ ] Cross-dataset testing
- [ ] Cross-language/domain testing where applicable
- [ ] Domain adaptation
- [ ] Multi-task learning
- [ ] Robustness testing

## Evaluation

- [ ] Accuracy
- [ ] Precision
- [ ] Recall
- [ ] Macro-F1
- [ ] Weighted-F1
- [ ] Confusion matrices
- [ ] Error analysis
- [ ] Explainability

## Application

- [ ] Inference pipeline
- [ ] Web interface
- [ ] Sentiment prediction
- [ ] Hate/offensive prediction
- [ ] Confidence scores
- [ ] Optional explanation

## Documentation

- [ ] Literature survey
- [ ] Methodology
- [ ] Experimental setup
- [ ] Results
- [ ] Error analysis
- [ ] Limitations
- [ ] Future work
- [ ] Dissertation
- [ ] Presentation
- [ ] Live demo

---

# 25. Recommended Development Order

Do **not** jump directly to the final multi-task architecture.

Follow this order:

```text
1. Audit datasets
       ↓
2. Understand labels
       ↓
3. Clean + standardize
       ↓
4. Build TF-IDF baselines
       ↓
5. Fine-tune IndicBERT on sentiment
       ↓
6. Fine-tune IndicBERT on hate/offensive data
       ↓
7. Combine compatible datasets
       ↓
8. Perform cross-dataset evaluation
       ↓
9. Try code-mixed/domain adaptation
       ↓
10. Build multi-task IndicBERT
       ↓
11. Compare all models
       ↓
12. Perform error analysis
       ↓
13. Add explainability
       ↓
14. Build application
       ↓
15. Final evaluation
       ↓
16. Dissertation + presentation
```

This progression ensures that **every stage has a usable result**, while also giving the capstone a clear research component beyond simply wrapping an existing GPT/LLM API.
