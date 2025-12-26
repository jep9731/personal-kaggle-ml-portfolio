# 📊 Machine Learning Portfolio – Kaggle & Applied Data Science Projects

This repository emphasizes **clear problem framing**, **exploratory analysis**, **modeling**, and **results**. Each project is self-contained, feel free to browse.

---

## 🚀 Projects Overview

| Project | Domain | Problem Type | Key Skills |
|:-------:|:-------:|:------------:|:-----------:|
| Stroke Prediction | Healthcare | Binary Classification | EDA, Imbalanced Data, XGBoost |
| Symptoms → Disease NLP | Healthcare / NLP | Multi-Class Classification | TF-IDF, NLP, ML Pipelines |
| House Prices (Kaggle) | Real Estate | Regression | Feature Engineering, Ensembles |
| Breast Cancer Diagnostic | Healthcare | Binary Classification | Model Evaluation, Scaling |
| Airline Delay Analysis | Transportation / Aviation Analytics | EDA & Time-Series Analysis | Large-Scale EDA, Time-Series Trends, Data Visualization |
| Exam Score Prediction (Kaggle) | Education / Analytics | Regression | EDA, Feature Engineering, Model Interpretation |

## 📂 Repository Structure
```
personal-kaggle-ml-portfolio/
│
├── README.md # Main portfolio landing page
│
├── projects/
│ ├── stroke-prediction/
│ │ ├── README.md # Project summary
│ │ ├── data/ # Dataset
│ │ │ └── stroke_prediction.R
│ │ │ └── healthcare-dataset-stroke-data.csv
│ │ ├── results/
│ │ | └── results.csv
│ │ ├── figures/
| | | └── logistic_roc.png
| | | └── rf_roc.png
| | | └── xgb_roc.png
| | | └── model_comparisons_roc.png
│ │ └── requirements/
| |   └── requirements.txt
│ │
│ ├── symptoms-disease-nlp/
│ │ ├── README.md
│ │ ├── data/
│ │ │ └── 01_text_eda.ipynb
│ │ │ └── 02_feature_engineering.ipynb
│ │ │ └── 03_multiclass_models.ipynb
│ │ └── results/
│ │   └── classification_report.txt
│ │   └── confusion_matrix.png
│ │
│ └── house-prices-regression/
│ | ├── README.md
│ | ├── data/
| | | └── Pasaye_Kaggle_competition.Rmd
| | | └── train.csv
| | | └── test.csv
│ | ├── submissions/
│ │ | └── submission.csv
│ | └── results/
| |   └── Pasaye_Kaggle_competition.html
│ |   └── rmse_scores.csv
│ |
| ├── breast-cancer-diagnostic/
│ │ ├── README.md
│ │ ├── data/
│ │ │ └── breast_cancer_prediction.Rmd
│ │ │ └── breast_cancer.csv
│ │ ├── figures/
| | | └── log_roc.png
| | | └── log_confusion_matrix.png
| | | └── svm_roc.png
| | | └── svm_confusion_matrix.png
| | | └── rf_roc.png
| | | └── rf_confusion_matrxi.png
| | | └── model_comparisons.png
│ │ ├── results/
│ │ | └── results.csv
| | └── requirements/
| |   └── requirements.txt
| | 
| ├── airline-delay-analysis/
│ │ ├── README.md
│ │ ├── data/
│ │ │ └── Airline_Delay_Cause.csv
| | ├── dashboard/
│ | | └──airline_delay_dashboard.Rmd
│ │ ├── notebooks/
│ │ | └── exploratory_analysis.R
| | └── requirements/
| |   └── requirements.txt
| |
| ├── exam-score-prediction/
│ │ ├── README.md
│ │ ├── data/
│ │ │ └── breast_cancer_prediction.Rmd
│ │ │ └── Exam_Score_Prediction.csv
│ │ ├── figures/
| | | └── log_roc.png
| | | └── log_confusion_matrix.png
| | | └── model_comparisons.png
│ │ ├── results/
│ │ | └── results.csv
| | └── requirements/
| |   └── requirements.txt
| |
└── .gitignore
```

---

# 🧩 Project Overview (Quick Read)

This section provides a high-level summary of each project.
Full technical details, notebooks, results, and evaluations are available inside each project’s folder.

## 🏡 House Prices – Advanced Regression (Kaggle)

**Problem:** Predict residential sale prices in Ames, Iowa

**Type:** Regression

**Domain:** Real Estate / Business Analytics

**Techniques Used**

* Feature engineering on 79 housing variables
* Handling missing data and skewed distributions
* Regularized linear models (Ridge, LASSO, ElasticNet)
* Tree-based ensemble models (Gradient Boosting, XGBoost)

**Evaluation Metric**
* RMSE on log-transformed SalePrice

**Highlight**

* Achieved strong cross-validated performance using XGBoost.
* Demonstrates Kaggle-style experimentation and model comparison.

`📂 projects/house-prices-regression/`

---

## 🧠 Stroke Prediction (Healthcare)

**Problem:** Predict stroke risk based on patient demographics and clinical factors

**Type:** Binary Classification

**Domain:** Healthcare Analytics

**Techniques Used**

* Missing value imputation
* Categorical encoding
* Class imbalance handling (class weights)
* Logistic Regression, Random Forest, XGBoost

**Evaluation Metrics**

* ROC-AUC
* Recall (stroke class)

**Highlight**

* Improved minority-class recall using logistic methods.
* Strong emphasis on healthcare-relevant evaluation metrics.

`📂 projects/stroke-prediction/`

---

## 🧬 Symptoms → Disease Classification (Medical NLP)

**Problem:** Predict diagnosed disease from reported patient symptoms

**Type:** Multi-Class Classification (30 diseases)

**Domain:** Medical NLP / Healthcare

**Techniques Used**

* Text preprocessing and normalization
* TF-IDF feature extraction
* Multinomial Naive Bayes, Logistic Regression, XGBoost

**Evaluation Metrics**

* Accuracy
* Macro F1-score

**Highlight**

* Effectively modeled symptom–disease relationships
* Demonstrates applied NLP and multi-class classification skills

`📂 projects/symptoms-disease-nlp/`

---

## 🧠 Breast cancer (Healthcare)

**Problem:** Classify breast tumors as **benign or malignant** cancer.

**Type:** Binary Classification

**Domain:** Healthcare Analytics

**Techniques Used**

* Missing value imputation
* Categorical encoding
* Class imbalance handling (SMOTE)
* Variable reduction using Principal Component Analysis (PCA)
* Logistic Regression, Random Forest, XGBoost

**Evaluation Metrics**

* ROC-AUC
* Recall (classification)
* Confusion matrix

**Highlight**

* Improved classification outcomes using ensemble methods.
* Strong emphasis on healthcare-relevant evaluation metrics.

`📂 projects/breast-cancer-diagnostic/`

---

## ✈️ Airline Delay Analysis – U.S. Aviation Operations EDA

**Problem:** Analyze causes and patterns of U.S. domestic airline delays across airlines, airports, and time

**Type:** Exploratory Data Analysis / Time-Series Analysis

**Domain:** Transportation Analytics / Aviation

**Techniques Used**

* Large-scale EDA on 20 years of airline delay data (2003–2022).
* Time-series analysis of delay trends by month and year.
* Aggregation and comparison across airlines and airports.
* Breakdown of delay causes (carrier, weather, NAS, security, late aircraft).
* Data visualization for operational and reliability insights.

**Key Insights Explored**

* Which airlines and airports are most delay-prone.
* How delay causes shift seasonally and over time.
* Dominant contributors to total delay minutes.
* Reliability comparisons across carriers and hubs.

**Highlight**

* Demonstrates real-world analytics on a 42MB operational dataset.
* Strong emphasis on storytelling, trends, and actionable aviation insights.

`📂 projects/airline-delay-analysis/`

---

## 🎓 Exam Score Prediction – Student Performance Analytics (Kaggle)

**Problem:** Predict student exam scores based on academic behavior, lifestyle habits, and learning environment factors

**Type:** Regression

**Domain:** Education Analytics / Behavioral Data Science

**Techniques Used**

* Exploratory data analysis on academic, behavioral, and lifestyle variables.
* Feature engineering (study habits, sleep patterns, attendance, exam conditions).
* Handling mixed data types (categorical + numerical features).
* Regression models (Linear Regression, Random Forest, Gradient Boosting, XGBoost).
* Model interpretation and feature importance analysis.

**Evaluation Metric**

* RMSE on exam score (0–100 scale)

**Highlight**

* Captures realistic, multi-factor influences on academic performance.
* Demonstrates end-to-end regression workflow with interpretable insights into student success drivers.

`📂 projects/exam-score-prediction/`
