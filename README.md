# AzgIntel

**AzgIntel** is an AI-powered natural language processing (NLP) platform designed for analyzing Armenian texts. It focuses on classifying text data, evaluating content, and generating actionable insights for research, AI monitoring, and automated text analysis.

---

## **Project Overview**

AzgIntel uses a **BERT-based multilingual model** to classify text data from Excel datasets. The pipeline includes:

- Data preprocessing and splitting into **Train / Validation / Test** sets
- Tokenization using `AutoTokenizer` from Hugging Face Transformers
- Dataset preparation with PyTorch `Dataset` and `DataLoader`
- Model training using **BERT for sequence classification**
- Evaluation using metrics like **accuracy, classification report, ROC-AUC**
- Fully reproducible in **Google Colab** or any Python environment with PyTorch & Transformers

---
