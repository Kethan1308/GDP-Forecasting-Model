# GDP Forecasting Using Machine Learning

## 📊 Project Overview

This project predicts the future Gross Domestic Product (GDP) of India using Machine Learning models.
It combines historical economic indicators such as GDP, inflation, imports, exports, unemployment rate, and population data to forecast GDP for upcoming years.

The project implements multiple prediction approaches and visualizes the results using interactive graphs.

---

## 🚀 Features

* Uses **Machine Learning models** for GDP prediction
* Combines multiple **economic indicators** for forecasting
* Implements **three prediction methods**:

  * Linear Regression
  * Random Forest (Year-only model)
  * Random Forest using multiple economic features
  * Random Forest using GDP change prediction
* Calculates **prediction accuracy using MAPE**
* Generates an **interactive Plotly visualization**
* Automatically creates fallback **sample datasets** if web scraping fails

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Plotly
* BeautifulSoup (Web Scraping)
* Requests

---

## 📂 Project Workflow

### 1. Data Collection

The project attempts to scrape data from economic websites such as:

* Worldometers (GDP data)
* WorldData (Inflation data)
* Other economic indicators

If scraping fails, the program automatically uses **sample datasets** to ensure the project runs successfully.

---

### 2. Data Processing

The collected data is:

* Merged from multiple sources
* Cleaned and formatted
* Stored into CSV datasets for modeling

---

### 3. Machine Learning Models

#### Method 1 – Year Based Prediction

Predicts GDP using only the **Year** as input.

Models used:

* Linear Regression
* Random Forest

---

#### Method 2 – Feature-Based Prediction

Predicts GDP using multiple economic indicators:

* GDP per capita
* Population
* Inflation
* Imports
* Exports
* Unemployment Rate

Random Forest is used for prediction.

---

#### Method 3 – GDP Change Prediction

Instead of predicting GDP directly, this model predicts **GDP change** using Random Forest and then calculates the future GDP.

This method is used to evaluate prediction accuracy.

---

## 📈 Visualization

The project generates an interactive Plotly graph showing:

* Past GDP values
* Random Forest predictions
* Linear Regression predictions
* Feature-based predictions
* GDP change-based predictions

The graph is saved as:

```
Fig12_Code_Output.html
```

---

## 📊 Accuracy Calculation

The model calculates prediction accuracy using **Mean Absolute Percentage Error (MAPE)**.

Example output:

* Actual GDP 2023: **$3.73 Trillion**
* Predicted GDP 2023
* MAPE
* Accuracy :99.54%

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install pandas numpy scikit-learn plotly requests beautifulsoup4
```

### 2️⃣ Run the Script

```bash
python gdp_forecasting.py
```

### 3️⃣ Output

* Console results showing predictions and accuracy
* Interactive graph
* HTML plot file

---

## 📊 Output Example

The final output graph visualizes GDP trends and predictions up to **2025**.

It includes:

* Historical GDP data
* Predictions from different machine learning models
* Accuracy annotation

---

## 📁 Repository Structure

```
GDP-Forecasting-ML
│
├── gdp_forecasting.py
├── data.csv
├── data1.csv
├── data2.csv
├── data3.csv
├── DataSet.csv
├── Fig12_Code_Output.html
└── README.md
```

---

## 🎯 Project Purpose

This project demonstrates how machine learning can be applied to **economic forecasting** by combining multiple macroeconomic indicators.

It highlights:

* Data collection
* Data preprocessing
* Machine learning modeling
* Prediction evaluation
* Data visualization

---

