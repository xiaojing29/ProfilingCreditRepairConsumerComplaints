# Profiling Credit Repair Complaints with Few-Shot Learning

This project applies few-shot learning (FSL) techniques to profile consumers based on their financial and legal knowledge, using complaint narratives related to credit reporting and credit repair scams from the Consumer Financial Protection Bureau (CFPB).

## Project Overview

- **Goal**: Classify consumers by their level of financial/legal knowledge using few-shot learning on real-world complaint narratives.
- **Context**: The rise of credit repair scams calls for tools to help identify and educate vulnerable consumers.
- **Approach**: Binary and 4-class classification tasks were explored using advanced NLP models and handcrafted features.

## Motivation

Traditional supervised learning requires large labeled datasets, which are often unavailable in rapidly evolving domains like finance. Few-shot learning offers a practical solution by training models with limited data. This project investigates its applicability to consumer complaint classification.

## Dataset

- **Source**: CFPB database (filtered for "credit repair" and "fraud or scam")
- **Raw entries**: 272 narratives
- **Final dataset sizes**:
  - **Binary**: 220 samples (0 = low knowledge, 1 = high knowledge)
  - **4-Class**: 218 samples (0 = ≤25%, 1 = 25–50%, 2 = 50–75%, 3 = ≥75%)
- **Annotation**: Initially manual, later refined after feedback
- **Balancing**: ChatGPT-generated examples were added using controlled prompts

## Methodology

### Feature Extraction

1. **Pre-trained Embeddings**:
   - Word2Vec
   - BERT
   - Sentence Transformers (using `[CLS]` and average embeddings)
2. **Manual Features**:
   - Average sentence length
   - Domain-specific terms
   - TF-IDF (via `TfidfVectorizer` from scikit-learn)

### Models

- **Word2Vec +**: KNN
- **BERT +**:
  - `BertForSequenceClassification`
  - Logistic Regression
  - MLPClassifier
- **Sentence Transformers +**:
  - Logistic Regression
  - MLPClassifier
  - LinearSVC

### Training Strategy

- Initial split: 70/15/15 → Revised: 10/80/10 (binary), 20/60/20 (4-class)
- Max epochs: 500
- Early stopping: patience = 10
- Optimizer: Adam

### Hyperparameter Tuning

- Manual tuning for BERT learning rates
- Grid search (`GridSearchCV`) for SVC and KNN

## Results Summary

| Task        | Best Model                          | Accuracy (%) | F1 Score (%) |
|-------------|-------------------------------------|--------------|--------------|
| Binary      | BERT + BertForSequenceClassification | 85.2         | 85.2         |
| 4-Class     | BERT + BertForSequenceClassification | 63.6         | 63.7         |

See the `results/` folder for:

- `accuracy_binary.png`, `accuracy_4_class.png`: Accuracy comparison charts
- `confusion_matrix_binary.png`, `confusion_matrix_4_class.png`: Confusion matrices

## Error Analysis

- Binary task: Tendency to misclassify high-knowledge cases as low-knowledge
- 4-class task: Difficulty distinguishing between adjacent knowledge levels
- Confusion matrices reveal systematic misclassifications across classes

## Repository Structure

CreditReportProfiling/
│
├── data/ # Official manually annotated data (2-class and 4-class)
│ ├── Complaints-official-2-classes.xlsx
│ └── Complaints-official-4-classes.xlsx
│
├── notebooks/ # All experiments and feature extraction done in Jupyter notebooks
│ ├── BERT+LR_MLPClassifier.ipynb
│ ├── BERT_ONLY.ipynb
│ ├── complaints_BERT+LR.ipynb
│ ├── complaints_BERT.ipynb
│ ├── Domain_terms.ipynb
│ ├── plots.ipynb
│ ├── STSB+LR_MLPClassifier.ipynb
│ ├── STSB + SVM.ipynb
│ └── STSB-BERT.ipynb
│
├── results/ # Evaluation plots and confusion matrices
│ ├── accuracy_4_class.png
│ ├── accuracy_binary.png
│ ├── confusion_matrix_4_class.png
│ └── confusion_matrix_binary.png
│
├── Project Report.pdf # Final project report with background, methods, and results
└── README.md # This file

## Run the Notebooks

Launch jupyter notebook

## References

Bansal et al. (2019), Fu et al. (2022), Villa-Cueva et al. (2023)
CFPB Consumer Complaints Database
Hugging Face Transformers, scikit-learn