# ✈️ Airline Delay Analysis – U.S. Aviation Operations EDA

## 📌 Project Overview

This project explores nearly two decades of U.S. domestic airline delay data to understand **why flights are delayed**, how delay causes vary across airlines and airports, and how patterns change over time.

Using large-scale operational data, the analysis focuses on delay **frequency**, **severity**, and **causes**, with an emphasis on clear visualizations and actionable insights for aviation analytics.

---

### 🧠 Problem Statement

**Analyze:**  
U.S. airline delays by carrier, airport, time, and delay cause  

**Answer questions such as:**
- Which airlines and airports experience the most delays?
- What delay causes dominate overall delay minutes?
- How do delay patterns change seasonally and over time?

---

### 📊 Dataset Description

- **Dataset:** Airline Delay Cause
- **Coverage:** ~2003–2022 (nearly 20 years)
- **File:** `Airline_Delay_Cause.csv` (~42 MB)
- **Granularity:** Monthly aggregates by:
  - Year & month
  - Operating carrier
  - Origin airport (IATA code + full name)

**Key Variables:**

- Total number of flights
- Number of delayed flights
- Delay minutes broken down by cause:
  - Carrier-related delays
  - Weather delays
  - NAS (air traffic system) delays
  - Security delays
  - Late aircraft delays

The dataset is well-suited for exploratory analysis, trend detection, and aviation-focused dashboards.

---

### 🔧 Techniques Used

#### Data Preparation
- Efficient loading and handling of large CSV files
- Aggregation by airline, airport, year, and month
- Handling missing values and zero-delay cases

#### Exploratory Data Analysis (EDA)
- Delay frequency vs. delay severity analysis
- Ranking airlines and airports by reliability
- Distribution analysis of delay causes
- Seasonal and long-term trend analysis

#### Time-Series Analysis
- Monthly and yearly delay trends
- Cause-specific trend comparisons
- Identification of structural changes over time

#### Visualization
- Time-series plots
- Comparative bar charts and rankings
- Cause breakdowns by airline and season

---

### 📈 Key Insights Explored

- Late aircraft and carrier-related issues account for the largest share of delay minutes
- Weather delays show strong seasonal patterns
- Certain hub airports consistently experience higher delay severity
- Airline reliability varies significantly when normalized by total flights

---

### ⭐ Project Highlights

- Demonstrates real-world EDA on a **large operational dataset**
- Strong emphasis on **data storytelling and trend analysis**
- Relevant for transportation analytics, operations research, and dashboarding
- Scales well to BI tools (Tableau, Power BI, Plotly)

---

### 🚀 Future Work

- Build predictive models for delay severity by airline and airport
- Create an interactive dashboard for delay exploration
- Normalize delay metrics by route volume
- Extend analysis to route-level or airport-pair insights

---

### 📎 Notes

This project focuses on **exploratory and descriptive analytics** rather than prediction.  
All analysis is reproducible using the provided notebooks.
