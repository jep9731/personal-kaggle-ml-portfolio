# 📊 Machine Learning Portfolio – Kaggle & Applied Data Science Projects

This repository emphasizes **clear problem framing**, **exploratory analysis**, **modeling**, and **results**. Each project is self-contained, feel free to browse.

---

## 📂 Repository Structure
```
kaggle-ml-portfolio/
│
├── README.md # Main portfolio landing page
│
├── projects/
│ ├── stroke-prediction/
│ │ ├── README.md # Project summary
│ │ ├── data/ # Dataset
│ │ ├── notebooks/
│ │ │ ├── 01_eda.ipynb
│ │ │ ├── 02_preprocessing.ipynb
│ │ │ └── 03_modeling.ipynb
│ │ └── results/
│ │ ├── metrics.json
│ │ └── figures/
│ │
│ ├── symptoms-disease-nlp/
│ │ ├── README.md
│ │ ├── data/
│ │ ├── notebooks/
│ │ │ ├── 01_text_eda.ipynb
│ │ │ ├── 02_feature_engineering.ipynb
│ │ │ └── 03_multiclass_models.ipynb
│ │ └── results/
│ │ ├── classification_report.txt
│ │ └── confusion_matrix.png
│ │
│ └── house-prices-regression/
│ ├── README.md
│ ├── data/
│ ├── notebooks/
│ │ ├── 01_eda.ipynb
│ │ ├── 02_feature_engineering.ipynb
│ │ ├── 03_models.ipynb
│ │ └── 04_ensemble.ipynb
│ ├── submissions/
│ │ └── submission_v1.csv
│ └── results/
│ └── rmse_scores.csv
│
├── shared_utils/ # Optional reusable helpers
│ ├── metrics.py
│ └── visualization.py
│
├── requirements.txt
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

* Improved minority-class recall using ensemble methods.
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
