# 📊 Mobile JKN Review Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.11-orange?style=flat&logo=tensorflow)](https://tensorflow.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2-blue?style=flat&logo=scikit-learn)](https://scikit-learn.org)
[![NLP](https://img.shields.io/badge/NLP-Natural%20Language%20Processing-green)]()

An end-to-end Machine Learning and Natural Language Processing (NLP) pipeline for scraping, processing, and analyzing public sentiment on user reviews of the **Mobile JKN BPJS Kesehatan** application on Google Play Store.

---

## 🎯 Overview

Mobile JKN is Indonesia's primary national health insurance mobile portal. This project extracts actionable user feedback by:
1. Scraping verified user reviews from Google Play Store.
2. Cleaning, normalizing, and tokenizing Indonesian informal language/slang.
3. Extracting semantic features via Doc2Vec / Paragraph Vector Distributed Bag-of-Words (PV-DBOW).
4. Training classification models to categorize reviews into **Positive** and **Negative** sentiments.
5. Visualizing dominant complaints, feature requests, and satisfaction metrics.

---

## 🔬 Pipeline Workflow

```
[Google Play Scraper] 
       │
       ▼
[Text Preprocessing & Normalization]
       │
       ▼
[Feature Extraction (PV-DBOW / Doc2Vec)]
       │
       ▼
[Classification Modeling & Evaluation]
       │
       ▼
[Insights & WordCloud Visualization]
```

### 1. Data Collection & Preprocessing
- **Source:** Google Play Store (`google-play-scraper`).
- **Text Normalization:** Case folding, punctuation removal, Indonesian slang dictionary mapping, Indonesian stopword filtering (`nltk`), and tokenization.

### 2. Feature Engineering & Modeling
- **Embedding:** Paragraph Vector Distributed Bag-of-Words (PV-DBOW) & TF-IDF vectorization.
- **Classifiers:** Deep Learning / Machine Learning classifiers evaluated across Accuracy, Precision, Recall, and F1-Score.

---

## 📁 Repository Structure

```
jkn_mobile_app_sentiment_analysis/
├── analisis_sentimen_aplikasi.ipynb       # Main EDA, modeling, and evaluation notebook
├── scrapping.ipynb                        # Play Store review harvesting script
├── inference_.ipynb                       # Live text inference & classification testing
├── perhitungan_manual_pvdbow.ipynb        # Mathematical formulation & step-by-step PV-DBOW
├── ulasan_aplikasi.csv                    # Raw collected review dataset
├── hasil_sentimen.csv                     # Labeled sentiment output dataset
└── requirements.txt.txt                   # Dependency manifest
```

---

## 🚀 Setup & Execution

```bash
git clone https://github.com/Akarptra/jkn_mobile_app_sentiment_analysis.git
cd jkn_mobile_app_sentiment_analysis
pip install -r requirements.txt.txt
jupyter notebook
```

---

## 👤 Author

- **Raka Putra Pratidina** — [GitHub (@Akarptra)](https://github.com/Akarptra)
