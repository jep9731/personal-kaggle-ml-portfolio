################################################################################
## Microsoft Stock - Advanced Time Series Analysis
## Extends: stock-time-series-analysis.py
## Covers:
##   1. STL Decomposition
##   2. ARIMA Forecasting
##   3. GARCH Volatility Modeling
##   4. Log Returns & Multivariate (Cross-correlation + Granger Causality)
##
## NOTE: Install any missing libraries with:
##   pip install statsmodels arch pandas matplotlib seaborn kagglehub
################################################################################

## ── IMPORTS ──────────────────────────────────────────────────────────────────

import os
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")  # Suppress convergence warnings during fitting

# STL Decomposition
from statsmodels.tsa.seasonal import STL

# ARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# GARCH
from arch import arch_model

# Granger Causality
from statsmodels.tsa.stattools import grangercausalitytests

## ── SETUP ────────────────────────────────────────────────────────────────────

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

## ── LOAD DATA ────────────────────────────────────────────────────────────────

path = kagglehub.dataset_download("vijayvvenkitesh/microsoft-stock-time-series-analysis")
df = pd.read_csv(os.path.join(path, "Microsoft_Stock.csv"))

# Parse dates and set as index — same as your original script
df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%Y %H:%M:%S")
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)  # Make sure data is in chronological order

print("Data loaded. Shape:", df.shape)
print(df.head())

################################################################################
## PART 1: STL DECOMPOSITION
################################################################################
#
# WHAT IS IT?
#   STL (Seasonal and Trend decomposition using Loess) splits your time series
#   into three separate components:
#       - Trend:     The long-term direction of the data (e.g. rising stock)
#       - Seasonal:  Repeating patterns over a fixed period (e.g. weekly cycles)
#       - Residual:  What's left over — random noise
#
# WHY DO IT?
#   Your ACF plot showed high autocorrelation, meaning the series has strong
#   memory. Before forecasting, it helps to see HOW MUCH of the movement is
#   structural (trend/seasonality) vs. random noise.
#
# PARAMETER: period=252
#   Stock markets have ~252 trading days per year, so we set the seasonal
#   period to 252 to detect annual patterns.
#
# HOW TO READ THE OUTPUT:
#   - If the trend line looks like your raw data → trend is dominant
#   - If seasonal spikes are large → seasonality matters
#   - If residuals look random/white-noise → your model is capturing the signal

print("\n" + "="*60)
print("PART 1: STL Decomposition")
print("="*60)

# Resample to monthly to make seasonal patterns clearer and computation faster
df_monthly = df['High'].resample('ME').mean()

# Fit the STL model
# period=12 for monthly data (12 months in a year)
stl = STL(df_monthly, period=12, robust=True)
# robust=True makes it less sensitive to outliers (e.g. COVID crash)

result = stl.fit()

# Plot all three components
fig = result.plot()
fig.set_size_inches(12, 8)
plt.suptitle("STL Decomposition of Monthly High Price", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "stl_decomposition.png"), dpi=300, bbox_inches="tight")
plt.show()

# INTERPRETATION HELPER: Print the strength of trend vs seasonality
# Values close to 1.0 = strong; close to 0.0 = weak
trend_strength = max(0, 1 - np.var(result.resid) / np.var(result.trend + result.resid))
seasonal_strength = max(0, 1 - np.var(result.resid) / np.var(result.seasonal + result.resid))
print(f"\nTrend Strength:    {trend_strength:.3f}  (1.0 = very strong trend)")
print(f"Seasonal Strength: {seasonal_strength:.3f}  (1.0 = very strong seasonality)")

################################################################################
## PART 2: ARIMA FORECASTING
################################################################################
#
# WHAT IS IT?
#   ARIMA = AutoRegressive Integrated Moving Average
#   It's a classic statistical model that predicts future values based on
#   past values and past errors. The three parameters are:
#       p (AR order):  How many past values to use
#       d (I order):   How many times to difference (you found d=1 already!)
#       q (MA order):  How many past forecast errors to use
#
# HOW TO PICK p AND q?
#   You use two plots together:
#       ACF  (Autocorrelation Function)  → helps pick q
#       PACF (Partial Autocorrelation)   → helps pick p
#
#   READING ACF:
#     - The lag where the bars drop inside the blue shaded band = your q value
#
#   READING PACF:
#     - The lag where the bars drop inside the blue shaded band = your p value
#
# COMMON STARTING POINT: ARIMA(1,1,1) — a safe first guess for stock data

print("\n" + "="*60)
print("PART 2: ARIMA Forecasting")
print("="*60)

# First, let's plot ACF and PACF on the DIFFERENCED series
# (differenced = stationary, which ARIMA needs)
high_diff = df['High'].diff().dropna()

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

plot_acf(high_diff, lags=40, ax=axes[0])
axes[0].set_title("ACF of Differenced High (use to pick q)")

plot_pacf(high_diff, lags=40, ax=axes[1], method='ywm')
axes[1].set_title("PACF of Differenced High (use to pick p)")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "acf_pacf_differenced.png"), dpi=300, bbox_inches="tight")
plt.show()

# ── FIT ARIMA MODEL ──────────────────────────────────────────────────────────
#
# We use the monthly resampled data to keep forecasting manageable.
# order=(1, 1, 1) means:
#   p=1: use 1 lag of past values
#   d=1: difference once (removes the trend — which you confirmed was needed)
#   q=1: use 1 lag of past errors

train = df_monthly[:-12]   # All data except last 12 months
test  = df_monthly[-12:]   # Hold out last 12 months to evaluate accuracy

model = ARIMA(train, order=(1, 1, 1))
fitted_model = model.fit()

print("\nARIMA Model Summary:")
print(fitted_model.summary())

# Forecast 12 steps (months) into the future
forecast = fitted_model.forecast(steps=12)

# Plot: actual vs forecast
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label="Training Data", color="blue")
plt.plot(test.index, test, label="Actual (held out)", color="green")
plt.plot(test.index, forecast, label="ARIMA Forecast", linestyle="--", color="red")
plt.xlabel("Date")
plt.ylabel("High Price")
plt.title("ARIMA(1,1,1) Forecast vs Actual — Monthly High")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "arima_forecast.png"), dpi=300, bbox_inches="tight")
plt.show()

# Evaluate accuracy using Mean Absolute Error (MAE) and RMSE
mae  = np.mean(np.abs(forecast.values - test.values))
rmse = np.sqrt(np.mean((forecast.values - test.values)**2))
print(f"\nForecast Accuracy (on held-out 12 months):")
print(f"  MAE  (Mean Absolute Error):       ${mae:.2f}")
print(f"  RMSE (Root Mean Squared Error):   ${rmse:.2f}")
print("  → Lower values = better. Compare to average price to gauge scale.")

################################################################################
## PART 3: GARCH VOLATILITY MODELING
################################################################################
#
# WHAT IS IT?
#   GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models
#   how VOLATILITY changes over time — not the price itself.
#
# WHY DO IT?
#   Your stationarity plot showed that the differenced series had larger spikes
#   around 2020 than in 2015–2018. This is called "volatility clustering":
#   big moves tend to follow big moves, calm periods follow calm periods.
#   GARCH captures this behavior.
#
# WHAT IT OUTPUTS:
#   A "conditional volatility" series — on any given day, how volatile was
#   the market? Peaks should align with major events (COVID crash, etc.)
#
# PARAMETERS: p=1, q=1 (same logic as ARIMA — a safe default)

print("\n" + "="*60)
print("PART 3: GARCH Volatility Modeling")
print("="*60)

# GARCH works on RETURNS, not raw prices
# We use log returns: log(price_today / price_yesterday) * 100
# Multiplying by 100 scales to percentage returns (helps GARCH converge)
log_returns = np.log(df['High'] / df['High'].shift(1)).dropna() * 100

print(f"\nLog Returns Summary:")
print(log_returns.describe())

# Plot log returns to visualize volatility clustering
plt.figure(figsize=(12, 4))
plt.plot(log_returns, color='navy', linewidth=0.5)
plt.axhline(0, color='red', linestyle='--', linewidth=0.8)
plt.title("Daily Log Returns of High Price (%) — Note clustering around 2020")
plt.xlabel("Date")
plt.ylabel("Log Return (%)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "log_returns.png"), dpi=300, bbox_inches="tight")
plt.show()

# Fit GARCH(1,1) model
# vol='Garch'  → standard GARCH model
# p=1, q=1     → 1 lag each for past variance and past squared returns
garch_model = arch_model(log_returns, vol='Garch', p=1, q=1, dist='normal')
garch_result = garch_model.fit(disp='off')  # disp='off' suppresses iteration output

print("\nGARCH Model Summary:")
print(garch_result.summary())

# Plot conditional volatility
cond_vol = garch_result.conditional_volatility

plt.figure(figsize=(12, 5))
plt.plot(cond_vol, color='darkorange', linewidth=0.8)
plt.title("GARCH(1,1) Conditional Volatility — Microsoft High Price")
plt.xlabel("Date")
plt.ylabel("Volatility (%)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "garch_volatility.png"), dpi=300, bbox_inches="tight")
plt.show()

print("\nKey observation: Look for peaks — they should align with the")
print("COVID crash (March 2020) and other major market events.")

################################################################################
## PART 4: LOG RETURNS & MULTIVARIATE ANALYSIS
################################################################################
#
# 4A — LOG RETURNS (why use them?)
#   Raw prices are non-stationary (you proved this with ADF).
#   Log returns ARE stationary and represent % change, which is what
#   analysts actually care about. They're also additive over time.
#
# 4B — CROSS-CORRELATION (does Volume lead price?)
#   We check if today's Volume is correlated with tomorrow's High.
#   A significant correlation at lag > 0 means Volume can PREDICT future price.
#
# 4C — GRANGER CAUSALITY TEST
#   "Does knowing X help predict Y, beyond what Y alone can tell us?"
#   We test: Does Volume Granger-cause the High price?
#   If p-value < 0.05 → YES, Volume adds predictive power

print("\n" + "="*60)
print("PART 4: Log Returns & Multivariate Analysis")
print("="*60)

# ── 4A: Log returns for all price columns ────────────────────────────────────
price_cols = ['Open', 'High', 'Low', 'Close']
log_ret_df = np.log(df[price_cols] / df[price_cols].shift(1)).dropna()
log_ret_df.columns = [c + '_logret' for c in price_cols]

print("\nLog Returns (first 5 rows):")
print(log_ret_df.head())

# Correlation heatmap of log returns
plt.figure(figsize=(7, 5))
sns.heatmap(log_ret_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation of Log Returns (Open/High/Low/Close)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "log_return_correlation.png"), dpi=300, bbox_inches="tight")
plt.show()

# ── 4B: Cross-correlation — Volume vs High ───────────────────────────────────
#
# We compute the cross-correlation between Volume and High log returns
# at multiple lags to see if volume movements PRECEDE price movements.

volume_logret = np.log(df['Volume'] / df['Volume'].shift(1)).dropna()
high_logret   = log_ret_df['High_logret']

# Align the two series (they may differ in length slightly)
aligned = pd.concat([volume_logret, high_logret], axis=1).dropna()
aligned.columns = ['Volume_logret', 'High_logret']

# Cross-correlations at lags 0–10
lags = range(0, 11)
cross_corrs = [aligned['Volume_logret'].corr(aligned['High_logret'].shift(-lag)) 
               for lag in lags]

plt.figure(figsize=(10, 4))
plt.bar(lags, cross_corrs, color='steelblue')
plt.axhline(0, color='black', linewidth=0.8)
plt.xlabel("Lag (days) — positive lag = Volume leads High")
plt.ylabel("Correlation")
plt.title("Cross-Correlation: Volume Log Return → Future High Log Return")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cross_correlation.png"), dpi=300, bbox_inches="tight")
plt.show()

# ── 4C: Granger Causality Test ───────────────────────────────────────────────
#
# Tests up to maxlag=5: "Does Volume log return help predict High log return?"
# Output: For each lag, you'll see an F-test and p-value.
# If p < 0.05 at any lag → Volume Granger-causes High at that lag

print("\nGranger Causality Test: Does Volume → High?")
print("(p < 0.05 = Volume adds predictive power for High at that lag)\n")

gc_data = aligned[['High_logret', 'Volume_logret']]  # Order matters: [Y, X]
gc_result = grangercausalitytests(gc_data, maxlag=5, verbose=True)


################################################################################
## SUMMARY & NEXT STEPS
################################################################################

print("\n" + "="*60)
print("ANALYSIS COMPLETE — Output files saved to:", output_dir)
print("="*60)
print("""
Files generated:
  stl_decomposition.png     → Trend / seasonal / residual breakdown
  acf_pacf_differenced.png  → Use to pick ARIMA p and q orders
  arima_forecast.png        → 12-month forecast vs actual
  log_returns.png           → Daily % changes showing volatility clustering
  garch_volatility.png      → Time-varying volatility (peaks = turbulence)
  log_return_correlation.png → How Open/High/Low/Close move together
  cross_correlation.png     → Does Volume predict future High?

Suggested next steps:
  1. Tune ARIMA: try different (p,d,q) combos and compare RMSE values
  2. Try auto_arima from pmdarima to automatically select the best order
  3. Try a SARIMA model if seasonal_strength > 0.3
  4. Try Facebook Prophet for a more modern forecasting approach (pip install prophet)
  5. Add exogenous variables (e.g. Volume) to an ARIMAX model
""")
