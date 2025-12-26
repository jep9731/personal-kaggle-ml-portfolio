# 🧬 Breast Cancer Diagnosis Prediction

**Domain:** Healthcare / Oncology  
**Problem Type:** Binary Classification

---

## 🔍 Business / Problem Context
Breast cancer is one of the most common cancers worldwide, and early diagnosis significantly improves treatment outcomes. Machine learning models can assist clinicians by providing reliable, data-driven predictions based on diagnostic measurements.

This project focuses on classifying breast tumors as **benign or malignant** using features computed from digitized images of fine needle aspirate (FNA) biopsies.

---

## 🎯 Objective
Develop and evaluate classification models that accurately predict whether a breast tumor is benign or malignant based on diagnostic features.

---

## 🧾 Dataset Overview
- **Records:** 569 patient samples  
- **Features:** 30 numeric features describing cell nuclei characteristics  
  - Radius, texture, perimeter, area, smoothness, concavity, symmetry, etc.
- **Target variable:** `diagnosis`
  - Malignant (M)
  - Benign (B)
- **Source:** UCI Machine Learning Repository

---

## 🛠 Methodology
- Exploratory Data Analysis to assess feature distributions and class balance
- Feature scaling and normalization
- Train–test split
- SMOTE (Synthetic Minority Oversampling Technique)
- Models evaluated:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest Classifier
- Performance evaluated using accuracy and ROC-AUC

---

## 📊 Results
| Model | Accuracy | ROC-AUC |
|------|----------|---------|
| Logistic Regression | **0.965** | 0.975 |
| SVM | 0.894 | **0.998** |
| Random Forest | **0.965** | **0.998** |

---

## 💡 Key Insights
- Mean radius, concavity, and texture features were highly predictive
- Proper feature scaling significantly improved linear model performance
- Ensemble models (i.e. **Random Forest**) achieved the strongest overall results

---

## ⚠️ Limitations & Next Steps
- Relatively small dataset limits generalization
- No model explainability included
- Future improvements:
  - SHAP or permutation feature importance
  - Threshold optimization for recall-focused screening use cases

---

## 🧠 Skills Demonstrated

- Binary classification
- Healthcare data analysis
- Model evaluation
- Feature scaling
