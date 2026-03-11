import requests
from bs4 import BeautifulSoup
import csv
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import plotly.graph_objects as go
import re
import numpy as np

print("="*60)
print("GDP FORECASTING PROJECT")
print("="*60)

# ==================== FALLBACK SAMPLE DATA ====================
print("\n[INFO] Creating sample data as fallback...")

sample_years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
sample_gdp = [1.68, 1.82, 1.83, 1.86, 2.04, 2.10, 2.29, 2.65, 2.70, 2.83, 2.67, 3.15, 3.39]
sample_gdp_per_cap = [1357, 1458, 1444, 1405, 1490, 1590, 1701, 1796, 1891, 1944, 1814, 1962, 2085]
sample_pop = [1.24e9, 1.26e9, 1.27e9, 1.29e9, 1.31e9, 1.32e9, 1.34e9, 1.35e9, 1.37e9, 1.38e9, 1.40e9, 1.41e9, 1.42e9]
sample_pop_change = [1.5, 1.5, 1.4, 1.3, 1.3, 1.2, 1.2, 1.2, 1.1, 1.1, 1.0, 0.9, 0.8]
sample_inflation = [10.5, 9.5, 9.3, 11.1, 6.7, 4.9, 4.95, 3.33, 3.94, 3.73, 6.62, 5.13, 6.7]
sample_imports = [450, 567, 571, 528, 529, 465, 480, 582, 640, 602, 510, 761, 911]
sample_exports = [375, 447, 448, 472, 468, 417, 440, 498, 539, 529, 500, 678, 760]
sample_unemployment = [8.3, 8.2, 8.1, 8.0, 8.0, 7.9, 7.8, 7.7, 7.7, 6.5, 10.2, 7.7, 7.3]

sample_df = pd.DataFrame({
    'Year': sample_years,
    'GDP ($)': [g * 1e12 for g in sample_gdp],
    'GDP per cap ($)': sample_gdp_per_cap,
    'Pop': sample_pop,
    'Pop. change': sample_pop_change,
    'Inflation (%)': sample_inflation,
    'Imports ($B)': sample_imports,
    'Exports ($B)': sample_exports,
    'Unemployment Rate (%)': sample_unemployment
})

sample_df.to_csv("data1.csv", index=False)
sample_df[['Year', 'Inflation (%)']].to_csv("data2.csv", index=False)
sample_df[['Year', 'Imports ($B)', 'Exports ($B)', 'Unemployment Rate (%)']].to_csv("data3.csv", index=False)
sample_df.to_csv("data.csv", index=False)
sample_df.to_csv("DataSet.csv", index=False)
print("[SUCCESS] Sample data created and saved to CSV files")

# ==================== ATTEMPT WEB SCRAPING ====================
print("\n[INFO] Attempting to scrape real data from websites...")

# (All scraping code as before – unchanged, but keep it for completeness)
# ... (I'm omitting the scraping blocks here for brevity, but you should keep them from the previous full code)

# ==================== MERGE AND CLEAN DATA ====================
print("\n[INFO] Merging and cleaning data...")

try:
    d1 = pd.read_csv("data1.csv")
    d2 = pd.read_csv("data2.csv")
    d3 = pd.read_csv("data3.csv")
    merged_df = pd.merge(d1, d2, on='Year', how='outer')
    merged_df = pd.merge(merged_df, d3, on='Year', how='outer')
    # Fix duplicate column names
    for col in merged_df.columns:
        if col.endswith('_x'):
            new_col = col.replace('_x', '')
            if new_col in merged_df.columns:
                merged_df[new_col] = merged_df[col]
                merged_df.drop(columns=[col], inplace=True)
        elif col.endswith('_y'):
            merged_df.drop(columns=[col], inplace=True)
    merged_df = merged_df[merged_df['Year'] >= 1993]
    merged_df = merged_df[merged_df['Year'] <= 2022]
    merged_df = merged_df.reset_index(drop=True)
    merged_df.to_csv("data.csv", index=False)
    print(f"[SUCCESS] Data merged successfully, {len(merged_df)} rows")
except Exception as e:
    print(f"[ERROR] Merging failed: {e}, using sample data")
    merged_df = sample_df.copy()
    merged_df.to_csv("data.csv", index=False)

def clean_value(value):
    if isinstance(value, str):
        try:
            if value.endswith('%'):
                return float(value[:-1])
            elif '$' in value:
                return float(value.replace('$', '').replace(',', ''))
            else:
                cleaned = float(value.replace(',', ''))
                return int(cleaned) if cleaned.is_integer() else cleaned
        except:
            return 0
    return value

try:
    with open('data.csv', newline='', encoding='utf-8') as input_file:
        with open('DataSet.csv', 'w', newline='', encoding='utf-8') as output_file:
            reader = csv.reader(input_file)
            writer = csv.writer(output_file)
            header = next(reader)
            writer.writerow(header)
            for row in reader:
                cleaned_row = [clean_value(val) for val in row]
                writer.writerow(cleaned_row)
    print("[SUCCESS] Data cleaned successfully")
except Exception as e:
    print(f"[ERROR] Data cleaning failed: {e}, using sample data")
    sample_df.to_csv("DataSet.csv", index=False)

# ==================== LOAD DATA FOR MODELING ====================
data = pd.read_csv("DataSet.csv")
required_cols = ['Year', 'GDP ($)', 'GDP per cap ($)', 'Pop', 'Inflation (%)', 'Imports ($B)', 'Exports ($B)', 'Unemployment Rate (%)']
if not all(col in data.columns for col in required_cols):
    print("[WARNING] Missing columns, using sample data")
    data = sample_df.copy()

print(f"\n[INFO] Loaded {len(data)} rows of data")

# ==================== PREDICTION MODELS ====================
print("\n" + "="*60)
print("RUNNING PREDICTION MODELS")
print("="*60)

# --- Method 1: Linear Regression and Random Forest (Year only) ---
X_year = data[['Year']].values
y_gdp = data['GDP ($)'].values

model_linear = LinearRegression()
model_rf = RandomForestRegressor(random_state=42, n_estimators=100)
model_linear.fit(X_year, y_gdp)
model_rf.fit(X_year, y_gdp)

years_pred = [2023, 2024, 2025]
linear_preds = [model_linear.predict([[y]])[0] for y in years_pred]
rf_preds = [model_rf.predict([[y]])[0] for y in years_pred]

predictions_df = pd.DataFrame({
    'Year': years_pred,
    'Linear Regression Predictions': [p/1e12 for p in linear_preds],
    'Random Forest Predictions': [p/1e12 for p in rf_preds]
})

print("\n[RESULTS] Method 1 - Year-only predictions (in Trillions USD):")
print(predictions_df.to_string(index=False))

# --- Method 2: Random Forest using all features ---
feature_cols = ['GDP per cap ($)', 'Pop', 'Inflation (%)', 'Imports ($B)', 'Exports ($B)', 'Unemployment Rate (%)']
X_features = data[feature_cols]
y_gdp = data['GDP ($)']

model_rf_features = RandomForestRegressor(random_state=42, n_estimators=100)
model_rf_features.fit(X_features, y_gdp)

# Predict future feature values using linear trend
future_features = []
for year in years_pred:
    feats = []
    for col in feature_cols:
        feat_model = LinearRegression()
        feat_model.fit(data[['Year']], data[col])
        feats.append(feat_model.predict([[year]])[0])
    future_features.append(feats)

gdp_pred_features = [model_rf_features.predict([f])[0] for f in future_features]
predicted_gdp_df_extended = pd.DataFrame({'Year': years_pred, 'GDP ($)': [p/1e12 for p in gdp_pred_features]})

print("\n[RESULTS] Method 2 - All features predictions (in Trillions USD):")
print(predicted_gdp_df_extended.to_string(index=False))

# --- Method 3: Random Forest using GDP change ---
data['GDP Change'] = data['GDP ($)'].diff().fillna(0)
X_change = data[feature_cols].iloc[1:]
y_change = data['GDP Change'].iloc[1:]

model_change = RandomForestRegressor(random_state=42, n_estimators=100)
model_change.fit(X_change, y_change)

current_gdp = data['GDP ($)'].iloc[-1] / 1e12
gdp_change_preds = []
for feats in future_features:
    change = model_change.predict([feats])[0] / 1e12
    current_gdp += change
    gdp_change_preds.append(current_gdp)

predicted_gdp_df1 = pd.DataFrame({'Year': years_pred, 'GDP ($)': gdp_change_preds})

print("\n[RESULTS] Method 3 - GDP change predictions (in Trillions USD):")
print(predicted_gdp_df1.to_string(index=False))

# ==================== ACCURACY CALCULATION ====================
gdp2023a = 3.73
gdp2023p = predicted_gdp_df1[predicted_gdp_df1['Year'] == 2023]['GDP ($)'].values[0]
mape = abs((gdp2023a - gdp2023p) / gdp2023a) * 100
accuracy = 100 - mape

print("\n" + "="*60)
print("ACCURACY CALCULATION")
print("="*60)
print(f"Actual GDP 2023: ${gdp2023a:.2f} Trillion")
print(f"Predicted GDP 2023 (Method 3): ${gdp2023p:.2f} Trillion")
print(f"MAPE: {mape:.2f}%")
print(f"Accuracy: {accuracy:.2f}%")

# ==================== PLOT - ORIGINAL FIG 12 COLORS ====================
print("\n" + "="*60)
print("CREATING FINAL PLOT (Original Fig 12 Colors)")
print("="*60)

fig = go.Figure()

# 1. Past GDP (blue)
fig.add_trace(go.Scatter(
    x=data['Year'],
    y=data['GDP ($)'] / 1e12,
    mode='lines+markers',
    name='Past GDP ($)',
    marker=dict(size=6, color='blue'),
    line=dict(color='blue', width=2)
))

# 2. Random Forest using gdp change (purple)
fig.add_trace(go.Scatter(
    x=predicted_gdp_df1['Year'],
    y=predicted_gdp_df1['GDP ($)'],
    mode='lines+markers',
    name='Random Forest using gdp change',
    marker=dict(size=6, color='purple'),
    line=dict(color='purple', width=2)
))

# 3. Random Forest Predictions (green)
fig.add_trace(go.Scatter(
    x=predictions_df['Year'],
    y=predictions_df['Random Forest Predictions'],
    mode='lines+markers',
    name='Random Forest Predictions',
    marker=dict(size=6, color='green'),
    line=dict(color='green', width=2)
))

# 4. Linear Regression Predictions (red)
fig.add_trace(go.Scatter(
    x=predictions_df['Year'],
    y=predictions_df['Linear Regression Predictions'],
    mode='lines+markers',
    name='Linear Regression Predictions',
    marker=dict(size=6, color='red'),
    line=dict(color='red', width=2)
))

# 5. Random Forest using all (yellow)
fig.add_trace(go.Scatter(
    x=predicted_gdp_df_extended['Year'],
    y=predicted_gdp_df_extended['GDP ($)'],
    mode='lines+markers',
    name='Random Forest using all',
    marker=dict(size=6, color='yellow'),
    line=dict(color='yellow', width=2)
))

# 6. 2023 GDP (dark blue marker)
fig.add_trace(go.Scatter(
    x=[2023],
    y=[3.73],
    mode='markers',
    name='2023 GDP',
    marker=dict(
        size=14,
        color='darkblue',
        line=dict(width=2, color='white')
    ),
    showlegend=True
))

# Layout – exactly as in document
fig.update_layout(
    title=dict(
        text='GDP Predictions',
        x=0.5,
        font=dict(size=16, family='Arial', color='black')
    ),
    xaxis_title='Year',
    yaxis_title='GDP (Trillion $)',
    xaxis=dict(
        range=[2010, 2026],
        dtick=2,
        tick0=2010,
        showgrid=True,
        gridcolor='lightgray'
    ),
    yaxis=dict(
        range=[1.5, 4.5],
        dtick=0.5,
        tick0=1.5,
        showgrid=True,
        gridcolor='lightgray'
    ),
    legend=dict(
        x=1.02,
        y=1,
        xanchor='left',
        yanchor='top',
        bgcolor='white',
        bordercolor='black',
        borderwidth=1
    ),
    margin=dict(l=80, r=150, t=80, b=80),
    template='plotly_white'
)

# Accuracy annotation (centered below plot)
fig.add_annotation(
    text=f"<b>Accuracy of Random Forest using GDP change of 2023: {accuracy:.2f}%</b>",
    xref="paper", yref="paper",
    x=0.5, y=-0.12,
    showarrow=False,
    font=dict(size=11, color='purple')
)

fig.show()
fig.write_html("Fig12_Code_Output.html")
print("[SUCCESS] Plot saved as 'Fig12_Code_Output.html'")

print("\n" + "="*60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("="*60)
