################################################################################
## Microsoft Stock – Core Time Series Analysis
## Author: Joshua Pasaye
## Dataset: Microsoft_Stock.csv (via Kaggle)
## Covers:
##   1. Data Loading & Cleaning
##   2. Trend Visualization
##   3. Monthly Resampling
##   4. Seasonality Detection (ACF)
##   5. Stationarity Testing (ADF)
##   6. Differencing to Achieve Stationarity
##   7. Moving Average Smoothing
##   8. ADF Test on Differenced Series
##
## NOTE: Install any missing libraries with:
##   pip install kagglehub pandas numpy matplotlib seaborn statsmodels
################################################################################

## ── IMPORTS ──────────────────────────────────────────────────────────────────

# API
import kagglehub
import os

# Data manipulation
import pandas as pd
import numpy as np

# Time-series
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

## ── SETUP ────────────────────────────────────────────────────────────────────

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
print(f"Directory '{output_dir}' ensured to exist.")

################################################################################
## PART 1: DATA LOADING & CLEANING
################################################################################
#
# WHAT IS IT?
#   We load the Microsoft Stock CSV from Kaggle and perform basic cleaning
#   before any analysis. This includes checking for missing values and
#   ensuring the Date column is in the correct datetime format.
#
# WHY DO IT?
#   Time series analysis is highly sensitive to missing data and
#   incorrectly typed date columns. Datetime formatting is essential
#   because pandas resampling and indexing require it.
#
# HOW TO READ THE OUTPUT:
#   - "No missing values" → proceed with analysis
#   - If missing values exist → imputation or removal is needed first

print("\n" + "="*60)
print("PART 1: Data Loading & Cleaning")
print("="*60)

## ── LOAD DATA ────────────────────────────────────────────────────────────────

path = kagglehub.dataset_download(
    "vijayvvenkitesh/microsoft-stock-time-series-analysis"
    )

df = pd.read_csv(os.path.join(path, "Microsoft_Stock.csv"))

print("\nSample of dataset:")
print(df.head())

## ── MISSING DATA ─────────────────────────────────────────────────────────────

if df.isnull().sum().all() == 0:
    print("\nNo missing values found. Proceed with analysis.")
else:
    print("\nMissing values detected. Proceed with imputation.")

## ── DATE FORMATTING ──────────────────────────────────────────────────────────
#
# Step 1: Parse the raw string into a datetime object so pandas understands it
# Step 2: Reformat into YYYY-MM-DD for clean display and indexing

df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%Y %H:%M:%S")
df['Date'] = df['Date'].dt.strftime("%Y-%m-%d")

print("\nDate column reformatted to YYYY-MM-DD.")
print(df['Date'].head())

################################################################################
## PART 2: TREND VISUALIZATION
################################################################################
#
# WHAT IS IT?
#   A line plot of the daily highest price (High) across the full date range.
#   This is the first and most important step in any time series analysis —
#   simply looking at the raw data.
#
# WHY DO IT?
#   Before applying any models, we need to visually assess:
#       - Is there a clear trend (upward, downward, flat)?
#       - Are there obvious spikes, crashes, or structural breaks?
#       - Does the variance appear stable or does it grow over time?
#
# HOW TO READ THE OUTPUT:
#   - A consistently rising line → upward trend (likely non-stationary)
#   - Variance expanding over time → heteroskedasticity (relevant for GARCH)
#   - Any sharp drops → external shocks (e.g. COVID-19 in early 2020)

print("\n" + "="*60)
print("PART 2: Trend Visualization")
print("="*60)

file_name = "high_plot.png"
full_path = os.path.join(output_dir, file_name)

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="Date", y="High")
plt.xlabel("Date")
plt.ylabel("High")
plt.title("Share Highest Price Over Time")
plt.savefig(full_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nPlot saved to: {full_path}")
print("INTERPRETATION: The chart shows MSFT's highest daily price over time.")
print("Look for a dominant upward trend, expanding variance, and any sudden")
print("price drops — all of which suggest the series is non-stationary.")

################################################################################
## PART 3: MONTHLY RESAMPLING
################################################################################
#
# WHAT IS IT?
#   Resampling compresses daily data into monthly averages. Each point on
#   the resampled chart represents the average High price for that month.
#
# WHY DO IT?
#   Daily stock data contains a lot of short-term noise that can obscure
#   the true long-run trend. Monthly resampling smooths that noise and
#   makes multi-year patterns much easier to see.
#
# PARAMETER: 'ME' = Month End frequency
#   pandas groups all trading days within each calendar month and
#   computes the mean of all numeric columns.
#
# HOW TO READ THE OUTPUT:
#   - A smoother version of the daily chart
#   - The overall direction (up/down/flat) becomes clearer
#   - Month-to-month dips are easier to identify (e.g. COVID crash)

print("\n" + "="*60)
print("PART 3: Monthly Resampling")
print("="*60)

# Convert Date back to datetime and set as index for resampling
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

df_resampled = df.resample('ME').mean(numeric_only=True)

print(f"\nResampled to {len(df_resampled)} monthly observations.")
print(df_resampled['High'].head())

file_name = "resample_plot.png"
full_path = os.path.join(output_dir, file_name)

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_resampled, x=df_resampled.index, y="High")
plt.xlabel("Date (Monthly)")
plt.ylabel("High")
plt.title("Monthly Resampling Highest Price Over Time")
plt.savefig(full_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nPlot saved to: {full_path}")
print("INTERPRETATION: Monthly averages confirm the long-run upward trend.")
print("The brief dip around early 2020 is the COVID-19 market shock.")

################################################################################
## PART 4: SEASONALITY DETECTION (ACF)
################################################################################
#
# WHAT IS IT?
#   The Autocorrelation Function (ACF) measures how correlated the series is
#   with its own past values (lags). Lag 1 = yesterday, Lag 5 = 5 days ago, etc.
#
# WHY DO IT?
#   If autocorrelation at many lags is high, the series has strong "memory" —
#   past values heavily influence future values. This is a key diagnostic
#   before choosing a forecasting model (e.g. ARIMA).
#
#   It also helps detect seasonality: if autocorrelation spikes at regular
#   intervals (e.g. every 5 lags = weekly pattern), seasonality is present.
#
# HOW TO READ THE OUTPUT:
#   - Bars all near 1.0 and slowly declining → strong trend, non-stationary
#   - Bars that cut off sharply → useful for picking MA order in ARIMA
#   - Bars oscillating at regular intervals → seasonal pattern present
#   - Bars within the blue shaded band → not statistically significant

print("\n" + "="*60)
print("PART 4: Seasonality Detection (ACF)")
print("="*60)

# Confirm Date is already the index
if "Date" not in df.columns:
    print("\n'Date' is already set as the index. Proceeding.")
else:
    df.set_index("Date", inplace=True)

file_name = "seasonality_plot.png"
full_path = os.path.join(output_dir, file_name)

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
plot_acf(df["High"], lags=40)
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.title("Autocorrelation Function (ACF) Plot")
plt.savefig(full_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nPlot saved to: {full_path}")
print("INTERPRETATION: If all bars are near 1.0 and decline slowly, the")
print("series is non-stationary with a dominant trend — not seasonal.")
print("This pattern motivates the ADF test and differencing in Parts 5 & 6.")

################################################################################
## PART 5: STATIONARITY TESTING (ADF TEST)
################################################################################
#
# WHAT IS IT?
#   The Augmented Dickey-Fuller (ADF) test formally tests whether a time
#   series is stationary. Stationarity means the mean, variance, and
#   autocorrelation structure do not change over time.
#
# WHY DO IT?
#   Most time series forecasting models (like ARIMA) assume the input
#   series is stationary. Running this test tells us whether we need to
#   transform the data (e.g. via differencing) before modeling.
#
# HYPOTHESES:
#   H0 (Null):       The series has a unit root → non-stationary
#   H1 (Alternative): The series is stationary
#
# HOW TO READ THE OUTPUT:
#   - p-value > 0.05 → fail to reject H0 → series is NON-STATIONARY
#   - p-value < 0.05 → reject H0 → series IS stationary
#   - ADF Statistic more negative than critical values → stronger evidence
#     of stationarity

print("\n" + "="*60)
print("PART 5: Stationarity Testing (ADF Test — Raw Series)")
print("="*60)

result = adfuller(df["High"])

print(f"\nADF Statistic:  {result[0]:.4f}")
print(f"p-value:        {result[1]:.6f}")
print("\nCritical Values:")
for key, val in result[4].items():
    print(f"   {key}: {val:.4f}")

print("\nINTERPRETATION:")
if result[1] > 0.05:
    print("  p-value > 0.05 → Fail to reject H0.")
    print("  The series is NON-STATIONARY. Differencing is required.")
else:
    print("  p-value < 0.05 → Reject H0.")
    print("  The series IS stationary. No differencing needed.")

################################################################################
## PART 6: DIFFERENCING TO ACHIEVE STATIONARITY
################################################################################
#
# WHAT IS IT?
#   Differencing replaces each value with the change from the previous value:
#       high_diff[t] = High[t] - High[t-1]
#   This removes the trend from the series, which is typically what makes
#   stock prices non-stationary.
#
# WHY DO IT?
#   Since the ADF test above confirmed non-stationarity, we apply first-order
#   differencing (d=1) to make the series stationary before modeling.
#   This is the "I" (Integrated) component in ARIMA(p, d, q).
#
# HOW TO READ THE OUTPUT:
#   - The original (blue) line shows the rising trend
#   - The differenced (green) line should fluctuate around zero with no trend
#   - If the differenced series still trends, try differencing a second time

print("\n" + "="*60)
print("PART 6: Differencing to Achieve Stationarity")
print("="*60)

df["high_diff"] = df["High"].diff()

print(f"\nFirst 5 values of differenced series (NaN at index 0 is expected):")
print(df["high_diff"].head())

file_name = "stationarity_plot.png"
full_path = os.path.join(output_dir, file_name)

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
plt.plot(df["High"], label="Original High", color="blue")
plt.plot(df["high_diff"], label="Differenced High", linestyle="--", color="green")
plt.legend()
plt.title("Original vs Differenced High")
plt.savefig(full_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nPlot saved to: {full_path}")
print("INTERPRETATION: The green differenced line should fluctuate around zero")
print("with no upward/downward trend. Notice higher variance around 2020 —")
print("this is volatility clustering, which GARCH can model.")

################################################################################
## PART 7: MOVING AVERAGE SMOOTHING
################################################################################
#
# WHAT IS IT?
#   A rolling (moving) average replaces each data point with the average of
#   the surrounding N observations. Here we use a window of 120 trading days
#   (~6 months), so each point reflects the average High over the prior 6 months.
#
# WHY DO IT?
#   Daily stock data is noisy. A moving average smooths short-term fluctuations
#   to reveal the underlying trend more clearly. It also serves as a simple
#   baseline "forecast" — if price is above the moving average, momentum is
#   positive; below it suggests a pullback.
#
# PARAMETER: window_size = 120 trading days ≈ 6 months
#   Shorter windows (e.g. 20) react faster but are noisier.
#   Longer windows (e.g. 200) are smoother but lag behind turns.
#
# HOW TO READ THE OUTPUT:
#   - When the blue line is above the orange → price is outpacing its trend
#   - When the blue line crosses below the orange → potential trend reversal
#   - The gap between them shows how far current price deviates from trend

print("\n" + "="*60)
print("PART 7: Moving Average Smoothing")
print("="*60)

window_size = 120
df["high_smoothed"] = df["High"].rolling(window=window_size).mean()

# Note: the first 119 values will be NaN because there aren't enough
# prior observations to fill the window yet
print(f"\n120-day moving average computed.")
print(f"First valid value at index {window_size - 1} (prior rows are NaN by design).")

file_name = "moving_avg_plot.png"
full_path = os.path.join(output_dir, file_name)

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
plt.plot(df["High"], label="Original High", color="blue")
plt.plot(df["high_smoothed"],
         label=f"Moving Average (Window={window_size})",
         linestyle="--", color="orange")
plt.xlabel("Date")
plt.ylabel("High")
plt.title("Original vs Moving Average")
plt.legend()
plt.savefig(full_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nPlot saved to: {full_path}")
print("INTERPRETATION: The orange moving average tracks the long-run trend.")
print("The blue daily price oscillates above and below it. The widening gap")
print("post-2019 suggests accelerating price growth beyond the trend baseline.")

################################################################################
## PART 8: ADF TEST ON DIFFERENCED SERIES
################################################################################
#
# WHAT IS IT?
#   We repeat the ADF test from Part 5, this time on the differenced series.
#   This confirms whether one round of differencing was sufficient to achieve
#   stationarity.
#
# WHY DO IT?
#   This is a critical validation step. If the differenced series is now
#   stationary (p < 0.05), we've confirmed that d=1 is the correct
#   integration order for ARIMA. If still non-stationary, we'd need d=2.
#
# HOW TO READ THE OUTPUT:
#   Same interpretation as Part 5 — but now we expect stationarity.
#   A much more negative ADF statistic and p-value near 0 is the goal.

print("\n" + "="*60)
print("PART 8: ADF Test on Differenced Series")
print("="*60)

# View the combined original and differenced data side-by-side
df_combined = pd.concat([df['High'], df['high_diff']], axis=1)
df_combined.columns = ['High', 'high_diff']

print("\nCombined dataset (first 5 rows):")
print(df_combined.head())
print("\nNote: high_diff row 0 is NaN — no prior value to difference against.")

# Drop the NaN before running ADF (it requires a complete series)
df.dropna(subset=["high_diff"], inplace=True)

result = adfuller(df["high_diff"])

print(f"\nADF Statistic:  {result[0]:.4f}")
print(f"p-value:        {result[1]:.6f}")
print("\nCritical Values:")
for key, val in result[4].items():
    print(f"   {key}: {val:.4f}")

print("\nINTERPRETATION:")
if result[1] < 0.05:
    print("  p-value < 0.05 → Reject H0.")
    print("  The DIFFERENCED series IS stationary. d=1 confirmed.")
    print("  This series is ready for ARIMA or other forecasting models.")
else:
    print("  p-value > 0.05 → Still non-stationary after first differencing.")
    print("  Consider second-order differencing (d=2).")

################################################################################
## SUMMARY
################################################################################

print("\n" + "="*60)
print("CORE ANALYSIS COMPLETE — Output files saved to:", output_dir)
print("="*60)
print("""
Files generated:
  high_plot.png          → Raw daily High price over time
  resample_plot.png      → Monthly average High price (trend smoothed)
  seasonality_plot.png   → ACF plot (checks for memory/seasonality)
  stationarity_plot.png  → Original vs differenced series
  moving_avg_plot.png    → 120-day moving average overlay

Key takeaways:
  - Strong upward trend confirmed visually and via ACF
  - ADF test confirmed non-stationarity in raw series (p > 0.05)
  - First differencing achieves stationarity (p < 0.05) → d=1
  - Volatility clustering visible around 2020 in differenced plot

Suggested next steps (see stock-time-series-advanced.py):
  1. STL Decomposition  → formally separate trend, seasonality, residual
  2. ARIMA Forecasting  → build on d=1 to forecast future prices
  3. GARCH Modeling     → model the volatility clustering seen in Part 6
  4. Multivariate       → test whether Volume predicts High (Granger Causality)
""")
