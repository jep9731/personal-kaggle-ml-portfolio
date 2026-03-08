# Symptoms → Disease Classification (Medical NLP)
### 📌 Symptoms to Disease Classification

**Domain:** Healthcare / Medical NLP

**Problem Type:** Multi-Class Classification (30 diseases)

---

## 🔍 Business / Problem Context

Healthcare decision support systems rely on accurate interpretation of patient-reported symptoms. Automating disease classification from symptom descriptions can support triage systems and clinical decision tools.

## 🎯 Objective

Develop a multi-class machine learning model that predicts a diagnosed disease based on a patient’s reported symptoms and demographic attributes.

## 🧾 Dataset Overview

* Records: 25,000 synthetic patient cases
* Features:
    * Age
    * Gender
    * Symptoms (3–7 symptoms, text)
    * Symptom count
* Target variable: Disease (30 classes)
* Data type: Fully synthetic (no real patient data)

## 🛠 Methodology

* Text cleaning and normalization
* Tokenization and TF-IDF vectorization of symptom text
* Combined text features with numeric attributes
* Models evaluated:
    * Multinomial Naive Bayes (baseline)
    * Logistic Regression (Softmax)
    * Random Forest
    * XGBoost

## 📊 Results

| Model | Accuracy | Macro F1 |
|:-----:|:--------:|:--------:|
| Naive Bayes | 0.78 | 0.75 |
| Logistic Regression | 0.84 | 0.82 |
| Random Forest | 0.85 | 0.83 |
| XGBoost | 0.88 | 0.86 |

## 💡 Key Insights

* TF-IDF captured meaningful symptom–disease patterns.
* Certain diseases (e.g., influenza vs. common cold) showed higher confusion.
* Ensemble models improved performance on minority classes.

## ⚠️ Limitations & Next Steps

* Synthetic data may not reflect real clinical complexity
* Symptoms treated as bag-of-words
* Future work: contextual embeddings (BERT), hierarchical disease grouping

## 🧠 Skills Demonstrated

* NLP feature engineering
* Multi-class classification
* Healthcare ML
