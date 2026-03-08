# 📈 Microsoft Stock – Time Series Analysis

**Domain:** Finance / Stock Market Analytics  
**Type:** Time Series Analysis  
**Language:** Python  
**Dataset:** [Microsoft Stock Time Series Analysis – Kaggle](https://www.kaggle.com/datasets/vijayvvenkitesh/microsoft-stock-time-series-analysis)

---

## 📋 Project Overview

This project performs an end-to-end time series analysis of Microsoft (MSFT) stock data spanning **2015–2021**. The analysis focuses on the daily highest price (`High`) and explores trend structure, stationarity, seasonality, volatility, and forecasting — building from foundational EDA toward advanced statistical modeling.

---

## 🎯 Objectives

- Visualize and understand long-term price trends
- Test for stationarity using the Augmented Dickey-Fuller (ADF) test
- Detect seasonality using Autocorrelation Function (ACF) analysis
- Decompose the series into trend, seasonal, and residual components (STL)
- Forecast future prices using ARIMA
- Model volatility clustering using GARCH
- Explore multivariate relationships between price and volume via Granger Causality

---

## 📁 Repository Structure

```
microsoft-stock-time-series/
│
├── README.md
│
├── scripts/
│   ├── stock-time-series-analysis.py       # Core EDA & stationarity analysis
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

---

## 🔍 Analysis Walkthrough

### 1. Exploratory Visualization
The raw `High` price chart shows a clear and sustained upward trend from ~$42 in 2015 to ~$245 by mid-2021, with notably accelerated growth post-2019.

### 2. Monthly Resampling
Resampling to monthly frequency smooths daily noise and confirms the long-run upward trajectory, with a visible dip and recovery in early 2020 (COVID-19 market shock).

### 3. Stationarity Testing (ADF)
The Augmented Dickey-Fuller test on the raw series produced a **p-value > 0.05**, confirming the series is **non-stationary**. First differencing (computing day-over-day changes) achieved stationarity, confirmed by a second ADF test with **p-value < 0.05**.

### 4. Autocorrelation (ACF) Analysis
The ACF plot shows autocorrelation values near **1.0** across all 40 lags, indicating strong persistence in the raw series — a hallmark of non-stationary, trend-driven data. This further motivates differencing before any modeling.

### 5. Moving Average Smoothing
A 120-day rolling average (approximately 6 months of trading days) was applied to highlight the underlying trend while suppressing short-term noise.

### 6. STL Decomposition
STL (Seasonal and Trend decomposition using Loess) separates the monthly series into:
- **Trend:** Captures the dominant multi-year growth story
- **Seasonal:** Tests for repeating annual patterns
- **Residual:** Remaining noise after trend and seasonality are removed

### 7. ARIMA Forecasting
Following the stationarity analysis (d=1 confirmed), PACF and ACF plots on the differenced series were used to identify the AR (`p`) and MA (`q`) orders. An `ARIMA(1,1,1)` model was fit on a training window and evaluated against a 12-month holdout set using MAE and RMSE.

### 8. GARCH Volatility Modeling
Log returns were computed and fed into a `GARCH(1,1)` model to capture **volatility clustering** — the tendency for turbulent periods to cluster together. The conditional volatility output clearly peaks around March 2020, aligning with the COVID-19 market crash.

### 9. Multivariate Analysis (Log Returns + Granger Causality)
Log returns were computed for Open, High, Low, and Close prices and visualized via a correlation heatmap. A Granger Causality test was applied to assess whether **Volume log returns** carry predictive signal for **High log returns** beyond the price series itself.

---

## 📊 Key Findings

| Analysis | Finding |
|---|---|
| ADF Test (raw) | Non-stationary (p > 0.05) — trend present |
| ADF Test (differenced) | Stationary (p < 0.05) — d=1 confirmed |
| ACF Plot | Near-perfect autocorrelation across all lags — strong persistence |
| STL | Trend dominates; seasonal component is relatively weak |
| ARIMA(1,1,1) | Reasonable 12-month forecast; RMSE evaluated on holdout |
| GARCH(1,1) | Clear volatility spike in March 2020 (COVID crash) |
| Granger Causality | Evaluated whether Volume leads High price at lags 1–5 |

---

## 🛠️ Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation and resampling |
| `numpy` | Numerical operations and log returns |
| `matplotlib` / `seaborn` | Visualization |
| `statsmodels` | ADF test, ACF/PACF, ARIMA, STL, Granger Causality |
| `arch` | GARCH volatility modeling |
| `kagglehub` | Dataset download |

Install all dependencies:
```bash
pip install pandas numpy matplotlib seaborn statsmodels arch kagglehub
```

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/jep9731/personal-kaggle-ml-portfolio.git
cd personal-kaggle-ml-portfolio/projects/microsoft-stock-time-series

# Install dependencies
pip install -r requirements/requirements.txt

# Run core analysis
python scripts/stock-time-series-analysis.py

# Run advanced modeling
python scripts/stock-time-series-advanced.py
```

> **Note:** A Kaggle API key is required for `kagglehub` to download the dataset.  
> Set it up via `~/.kaggle/kaggle.json` — see [Kaggle API docs](https://www.kaggle.com/docs/api).

---

## 📌 Suggested Next Steps

- Tune ARIMA order using `auto_arima` from `pmdarima` for automated selection
- Explore SARIMA if seasonal strength is significant
- Try Facebook Prophet for a modern, interpretable forecasting alternative
- Add Volume as an exogenous variable in an ARIMAX model
- Extend the analysis to Open, Low, Close, and Volume columns

---

*Part of the [Personal Kaggle ML Portfolio](https://github.com/jep9731/personal-kaggle-ml-portfolio)*
