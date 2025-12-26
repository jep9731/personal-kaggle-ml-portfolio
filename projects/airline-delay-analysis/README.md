# ✈️ Airline Delay Analysis – U.S. Aviation Operations  
### EDA & Interactive Dashboarding with R (flexdashboard)

### 📌 Project Overview

This project analyzes nearly two decades of U.S. domestic airline delay data to understand **why flights are delayed**, how delay causes vary across airlines and airports, and how these patterns change over time.

The analysis culminates in an **interactive dashboard built in R using `flexdashboard`**, designed for exploratory analysis, stakeholder reporting, and operational insight.

---

### 🧠 Problem Statement

**Analyze and communicate:**
- Airline and airport reliability
- Delay severity and frequency
- Root causes of delays across time and seasonality

**Primary objective:**  
Enable users to **interactively explore delay patterns** by airline, airport, time period, and delay cause.

---

### 📊 Dataset Description

- **Dataset:** Airline Delay Cause
- **Coverage:** ~2003–2022
- **File:** `Airline_Delay_Cause.csv` (~42 MB)
- **Granularity:** Monthly aggregates by:
  - Year & month
  - Operating carrier
  - Origin airport (IATA code + full airport name)

#### Delay Cause Categories

- Carrier-related delays  
- Weather delays  
- NAS (air traffic system) delays  
- Security delays  
- Late aircraft delays  

The dataset’s structure is well-suited for **time-series analysis and interactive filtering**.

---

### 🔧 Tools & Technologies

- **R**
- **tidyverse** (data wrangling & EDA)
- **lubridate** (time handling)
- **flexdashboard** (dashboard layout)
- **ggplot2 / plotly** (static & interactive visualizations)
- **DT** (interactive tables)

---

### 🔬 Analysis Workflow

#### Data Preparation
- Efficient loading of large CSV files
- Aggregation by airline, airport, year, and month
- Creation of derived metrics:
  - Delay rate
  - Average delay minutes per flight
  - Cause-level delay shares

#### Exploratory Data Analysis (EDA)
- Airline and airport reliability rankings
- Delay frequency vs severity analysis
- Cause-level contribution analysis
- Seasonal and long-term trend exploration

---

### 📊 Dashboard Design (flexdashboard)

The dashboard is built using **`flexdashboard`** to support interactive, self-service analysis.

### Dashboard Sections

**1. Overview**
- Total flights vs delayed flights
- Average delay minutes
- High-level cause breakdown

**2. Airline Performance**
- Delay rate and severity by carrier
- Cause distribution per airline
- Airline reliability rankings

**3. Airport Performance**
- Delay severity by airport
- Hub vs non-hub comparisons
- Seasonal airport delay patterns

**4. Time & Seasonality**
- Monthly and yearly delay trends
- Cause-specific time-series views
- Weather vs operational delay comparisons

#### Interactivity Features

- Dropdown filters (airline, airport, year)
- Hover-enabled charts (via `plotly`)
- Sortable and searchable tables (`DT`)
- Responsive layout for wide and narrow screens

---

### 📈 Key Insights Explored

- Late aircraft and carrier-related delays dominate total delay minutes
- Weather delays exhibit strong seasonal patterns
- Major hub airports show higher delay severity
- Airline reliability varies significantly when normalized by flight volume

---

### ⭐ Project Highlights

- End-to-end analytics project using **R + flexdashboard**
- Designed for **exploration, reporting, and storytelling**
- Handles large, real-world operational data
- Strong fit for data analyst and BI-focused roles

---

### 🚀 Future Enhancements

- Add delay forecasting using time-series models
- Publish dashboard via R Markdown / Shiny Server
- Add route-level analysis
- Optimize performance with data.table or Arrow

---

### 📎 Notes

This project focuses on **exploratory and descriptive analytics** rather than prediction.  
All insights are reproducible using the provided R scripts and dashboard code.
