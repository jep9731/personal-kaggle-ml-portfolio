# 🎓 Exam Score Prediction – Student Performance Analytics

### 📌 Project Overview

This project analyzes and predicts student exam scores based on a wide range of academic, behavioral, lifestyle, and environmental factors.  
The goal is to understand **what drives academic performance** and build regression models that can accurately estimate exam outcomes on a 0–100 scale.

The dataset is designed to mimic real-world educational settings, making this project suitable for applied machine learning, education analytics, and behavioral data science.

---

### 🧠 Problem Statement

**Predict:**  
* A student’s final exam score (0–100)

**Using:**  
* Academic habits
* Lifestyle routines
* Study environment
* Exam conditions

---

### 📊 Dataset Description

- **Source:** Kaggle – *Exam Score Prediction Dataset*
- **Size:** 20,000 student records
- **Target Variable:** Exam Score
- **Feature Categories:**
  - Academic behavior (study hours, attendance)
  - Lifestyle factors (sleep duration, sleep quality)
  - Learning environment (internet access, facilities rating)
  - Study methods and exam difficulty
  - Behavioral and institutional variables

The target score is generated using a weighted formula that reflects realistic academic performance patterns.

---

### 🔧 Techniques Used

#### Exploratory Data Analysis (EDA)
- Distribution analysis of exam scores
- Correlation analysis between lifestyle, academic, and environmental factors
- Identification of key performance drivers

#### Feature Engineering
- Encoding categorical variables
- Aggregating and transforming behavioral indicators
- Scaling numerical features where appropriate

#### Modeling
- Linear Regression (baseline)
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

#### Model Interpretation
- Feature importance analysis
- Comparison of academic vs lifestyle influence on performance

---

### 📏 Evaluation Metric

- **RMSE (Root Mean Squared Error)** on predicted exam scores

RMSE was chosen to penalize larger prediction errors and maintain interpretability on the original score scale.

---

### ⭐ Key Highlights

- Demonstrates a full **end-to-end regression pipeline** on mixed-type data.
- Reveals the relative importance of **study habits, sleep, and attendance**.
- Emphasizes interpretability alongside predictive performance.
- Suitable for education-focused analytics and behavioral ML applications.
