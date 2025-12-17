# House Prices: Advanced Regression Techniques (Kaggle)
### 📌 House Prices Prediction

**Domain:** Real Estate / Business Analytics

**Problem Type:** Regression

**Platform:** Kaggle Competition

---

## 🔍 Business / Problem Context

Accurate house price estimation is critical for buyers, sellers, and real estate professionals. Beyond obvious factors like size and location, housing prices are influenced by complex interactions between structural quality, neighborhood characteristics, and temporal effects.

This project uses the Ames Housing dataset, a rich, real-world dataset with detailed property attributes, to build predictive models that estimate residential sale prices.

## 🎯 Objective

Predict the final sale price of homes in Ames, Iowa, using advanced regression techniques and feature engineering, and generate Kaggle-ready submission files evaluated on log-RMSE.

## 🧾 Dataset Overview

* Training records: 1,460 houses
* Test records: 1,459 houses
* Features: 79 explanatory variables
    * Property size and layout
    * Construction quality and condition
    * Neighborhood and zoning information
    * Garage, basement, and exterior features
* Target variable: SalePrice
* Data source: Ames Housing Dataset (Dean De Cock)

## 📏 Evaluation Metric

**Root Mean Squared Error (RMSE)** on the logarithm of `SalePrice`.
  * Log transformation penalizes relative errors equally.
  * Encourages models to perform well on both low- and high-priced homes.

## 🛠 Methodology

1. Exploratory Data Analysis (EDA)
    * Analyzed missing values and feature distributions.
    * Identified skewed numeric variables.
    * Explored correlations between features and sale price.
2. Data Preprocessing
    * Imputed missing values based on domain logic (e.g., "None" vs median).
    * Log-transformed skewed numerical features.
    * One-hot encoded categorical variables.
3. Feature Engineering
    * Created aggregated quality and area features.
    * Removed low-variance and redundant variables.
    * Reduced multicollinearity for linear models.
4. Modeling
Models trained and compared:
    * Linear Regression (baseline)
    * LASSO Regression
    * Random Forest Regressor
    * Gradient Boosting Regressor
    * XGBoost Regressor

Cross-validation was used to ensure robust performance.

## 📊 Results

| Model | CV RMSE (log) |
|:-----:|:-------------:|
| Linear Regression | 0.1697500 |
| LASSO | 0.1694880 |
| Random Forest | 0.1698521 |
| Gradient Boosting | 0.1737498 |
| XGBoost | 0.1767609 | 

**Best Model:** LASSO Regressor

## 📤 Kaggle Submission

Generated submission files in the required format:
```
Id,SalePrice
1461,169000.1
1462,187724.12
```
Submission files stored in the `submissions/` directory

## 💡 Key Insights

* Overall quality and living area were the strongest predictors.
* Log transformation significantly improved linear model performance.
* Tree-based ensemble models captured non-linear relationships effectively.

## ⚠️ Limitations & Next Steps

* Feature interactions could be more data driven
* Hyperparameter tuning could be further optimized
* Future work:
    * Stacking / blending models
    * SHAP-based feature importance
    * Automated feature selection

## 🧠 Skills Demonstrated

* Regression modeling
* Feature engineering
* Cross-validation
* Kaggle competition workflow
* Business-oriented evaluation
