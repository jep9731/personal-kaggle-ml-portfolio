# Stroke Prediction (Healthcare Classification)
### 📌 Stroke Prediction

**Domain:** Healthcare

**Problem Type:** Binary Classification

---

## 🔍 Business / Problem Context

Stroke is the second leading cause of death globally. Early identification of patients at higher risk enables preventive interventions and better allocation of healthcare resources.

## 🎯 Objective

Build a machine learning model to predict whether a patient is likely to experience a stroke based on demographic, clinical, and lifestyle factors.

## 🧾 Dataset Overview

* Records: ~5,000 patients
* Features:
    * Demographics: age, gender, residence type
    * Medical history: hypertension, heart disease
    * Clinical: average glucose level, BMI
    * Lifestyle: smoking status
* Target variable: stroke (1 = stroke, 0 = no stroke)
* Source: Educational healthcare dataset (confidential)

## 🛠 Methodology

* Exploratory Data Analysis to assess class imbalance and feature distributions
* Missing value imputation for BMI
* One-hot encoding for categorical variables
* Addressed class imbalance using class weighting
* Models trained and evaluated:
    * Logistic Regression (baseline)
    * Random Forest
    * XGBoost

## 📊 Results

| Model | ROC-AUC | Recall (Stroke) |
|:-----:|:-------:|:---------------:|
|Logistic Regression | **0.86** | **0.95** |
| Random Forest | 0.78 | 0.36 |
| XGBoost | 0.82 | 0.40 |

**Logistic Regression** performed the best in predicting stroke.

## 💡 Key Insights

* Age and average glucose level were the strongest predictors.
* Severe class imbalance required recall-focused evaluation.
* Linear baseline outperformed tree-based models.

## ⚠️ Limitations & Next Steps

* Dataset size limits generalization.
* No longitudinal patient data.
* Future work:
   * Calibration
   * SHAP explainability

## 🧠 Skills Demonstrated

* Healthcare data preprocessing
* Imbalanced classification
* Model evaluation
