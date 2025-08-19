from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
# 📥 Load and clean dataset
try:
    df = pd.read_excel('AirQualityUCI.xlsx')
except FileNotFoundError:
    raise Exception("❌ 'AirQualityUCI.xlsx' not found. Place it in the same folder as app.py.")

df = df.dropna()
df = df.select_dtypes(include='number')

# ✅ Use correct benzene column name
target = 'C6H6(GT)'
if target not in df.columns:
    raise Exception(f"❌ '{target}' column not found in dataset.")

# 🎯 Select 8 important features (based on correlation or domain knowledge)
features = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)',
            'PT08.S5(O3)', 'NOx(GT)', 'NO2(GT)', 'T']

# 🧠 Train and save model + scaler if not already saved
if not os.path.exists('rf_model.pkl') or not os.path.exists('scaler.pkl'):
    print("🔄 Training model and saving .pkl files...")
    X = df[features]
    y = df[target]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestRegressor()
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, open('rf_model.pkl', 'wb'))
    joblib.dump(scaler, open('scaler.pkl', 'wb'))

    print("✅ Pickle files created: rf_model.pkl, scaler.pkl")
else:
    print("✅ Pickle files already exist.")

# 🔄 Load model and scaler
model = joblib.load(open('rf_model.pkl', 'rb'))
scaler = joblib.load(open('scaler.pkl', 'rb'))
