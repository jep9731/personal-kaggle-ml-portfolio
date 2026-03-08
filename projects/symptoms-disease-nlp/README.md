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

## 📁 Repository Structure

```
symptoms-disease-nlp/
│
├── README.md
│
├── scripts/
│   ├── symptoms-disease-nlp.py             # Core EDA & stationarity analysis
│   └── stock-time-series-advanced.py       # STL, ARIMA, GARCH, multivariate
│
├── figures/
│   ├── high_plot.png                        # Raw highest price over time
│   ├── resample_plot.png                    # Monthly resampled trend
│   ├── seasonality_plot.png                 # ACF plot
│   ├── stationarity_plot.png                # Original vs differenced series
│   ├── moving_avg_plot.png                  # 120-day moving average
│   ├── stl_decomposition.png                # STL trend/seasonal/residual
│   ├── acf_pacf_differenced.png             # ACF + PACF for ARIMA order selection
│   ├── arima_forecast.png                   # 12-month forecast vs actual
│   ├── log_returns.png                      # Daily log return series
│   ├── garch_volatility.png                 # Conditional volatility over time
│   └── log_return_correlation.png           # Cross-column correlation heatmap
│
└── requirements/
    └── requirements.txt
```

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
