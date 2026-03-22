# 📊 Machine Learning Portfolio – Kaggle & Applied Data Science Projects

This repository emphasizes **clear problem framing**, **exploratory analysis**, **modeling**, and **results**. Each project is self-contained, feel free to browse.

---

## 🚀 Projects Overview

| Project | Domain | Problem Type | Key Skills |
|:-------:|:-------:|:------------:|:-----------:|
| Stroke Prediction | Healthcare | Binary Classification | EDA, Imbalanced Data, XGBoost |
| House Prices | Real Estate | Regression | Feature Engineering, Ensembles |
| Breast Cancer Diagnostic | Healthcare | Binary Classification | Model Evaluation, Scaling |
| Airline Delay Analysis | Transportation / Aviation Analytics | EDA & Time-Series Analysis | Large-Scale EDA, Time-Series Trends, Data Visualization |
| Exam Score Prediction | Education / Analytics | Regression | EDA, Feature Engineering, Model Interpretation |
| Microsoft Stock Analysis | Finance / Stock Market | Time Series Analysis | STL, ARIMA, GARCH, Stationarity, Volatility Modeling |
| Time Series Analysis | Finance | Time Series Analysis | ARIMA, GARCH, Stationarity, Volatility Modeling |

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
│ │ │ └── exam_score_prediction.Rmd
│ │ │ └── Exam_Score_Prediction.csv
│ │ ├── results/
│ │ | └── results.csv
| | └── requirements/
| |   └── requirements.txt
| |
| ├── microsoft-stock-time-series/
│ │ ├── README.md
│ │ ├── scripts/
│ │ │ └── stock-time-series-analysis.py
│ │ │ └── stock-time-series-advanced.py
│ │ ├── figures/
| | | └── high_plot.png
| | | └── resample_plot.png
| | | └── seasonality_plot.png
| | | └── stationarity_plot.png
| | | └── moving_avg_plot.png
| | | └── stl_decomposition.png
| | | └── acf_pacf_differenced.png
| | | └── arima_forecast.png
| | | └── log_returns.png
| | | └── garch_volatility.png
| | | └── log_return_correlation.png
| | └── requirements/
| |   └── requirements.txt
| |
| ├── time-series-analysis/
│ │ ├── README.md
│ │ ├── scripts/
│ │ │ └── stock-time-series-analysis.py
│ │ │ └── stock-time-series-advanced.py
│ │ ├── figures/
| | | └── high_plot.png
| | | └── resample_plot.png
| | | └── seasonality_plot.png
| | | └── stationarity_plot.png
| | | └── moving_avg_plot.png
| | | └── stl_decomposition.png
| | | └── acf_pacf_differenced.png
| | | └── arima_forecast.png
| | | └── log_returns.png
| | | └── garch_volatility.png
| | | └── log_return_correlation.png
| | └── requirements/
| |   └── requirements.txt
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

* Improved classification outcomes using Logistical Regression.
* Strong emphasis on healthcare-relevant evaluation metrics.

`📂 projects/breast-cancer-diagnostic/`

---

## ✈️ Airline Delay Analysis – U.S. Aviation Operations EDA - In progress

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

---

## 📈 Microsoft Stock – Time Series Analysis

**Problem:** Analyze and model Microsoft (MSFT) daily stock price behavior from 2015 to 2021

**Type:** Time Series Analysis

**Domain:** Finance / Stock Market Analytics

**Techniques Used**

* Trend visualization and monthly resampling of daily OHLCV data
* Stationarity testing using the Augmented Dickey-Fuller (ADF) test
* First differencing to achieve stationarity (d=1 confirmed)
* Autocorrelation (ACF) and Partial Autocorrelation (PACF) analysis
* 120-day moving average smoothing
* STL decomposition into trend, seasonal, and residual components
* ARIMA(1,1,1) forecasting with 12-month holdout evaluation
* GARCH(1,1) volatility modeling to capture volatility clustering
* Log return computation and cross-column correlation analysis
* Granger Causality test (Volume → High price)

**Evaluation Metrics**

* ADF test statistic and p-value (stationarity)
* MAE and RMSE on held-out 12-month forecast window (ARIMA)
* Conditional volatility series (GARCH)

**Key Findings**

* Raw price series is non-stationary; first differencing achieves stationarity
* ACF shows near-perfect persistence across all lags — trend dominates
* STL confirms trend as the primary structural component
* GARCH conditional volatility peaks sharply in March 2020 (COVID crash)
* Granger Causality evaluated at lags 1–5 for Volume → High predictive power

**Highlight**

* Covers the full time series workflow from EDA to forecasting to volatility modeling.
* Bridges classical statistical methods (ARIMA, ADF) with financial risk modeling (GARCH).
* COVID-19 market shock is clearly visible and captured across multiple analyses.

`📂 projects/microsoft-stock-time-series/`

---

## 📈 Time Series Analysis

**Problem:** TBF

**Type:** Time Series Analysis

**Domain:** Finance / Stock Market Analytics

**Techniques Used**

* Trend visualization and monthly resampling of daily OHLCV data
* Stationarity testing using the Augmented Dickey-Fuller (ADF) test
* First differencing to achieve stationarity (d=1 confirmed)
* Autocorrelation (ACF) and Partial Autocorrelation (PACF) analysis
* 120-day moving average smoothing
* STL decomposition into trend, seasonal, and residual components
* ARIMA(1,1,1) forecasting with 12-month holdout evaluation
* GARCH(1,1) volatility modeling to capture volatility clustering
* Log return computation and cross-column correlation analysis
* Granger Causality test (Volume → High price)

**Evaluation Metrics**

* ADF test statistic and p-value (stationarity)
* MAE and RMSE on held-out 12-month forecast window (ARIMA)
* Conditional volatility series (GARCH)

**Key Findings**

* Raw price series is non-stationary; first differencing achieves stationarity
* ACF shows near-perfect persistence across all lags — trend dominates
* STL confirms trend as the primary structural component
* GARCH conditional volatility peaks sharply in March 2020 (COVID crash)
* Granger Causality evaluated at lags 1–5 for Volume → High predictive power

**Highlight**

* Covers the full time series workflow from EDA to forecasting to volatility modeling.
* Bridges classical statistical methods (ARIMA, ADF) with financial risk modeling (GARCH).
* COVID-19 market shock is clearly visible and captured across multiple analyses.

`📂 projects/time-series-analysis-ml/`
