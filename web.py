from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# ✅ Load model and scaler with joblib
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    p1 = float(request.form['PT08_S1_CO'])
    p2 = float(request.form['PT08_S2_NMHC'])
    p3 = float(request.form['PT08_S3_NOx'])
    p4 = float(request.form['PT08_S4_NO2'])
    p5 = float(request.form['PT08_S5_O3'])
    p6 = float(request.form['NOx_GT'])
    p7 = float(request.form['NO2_GT'])
    p8 = float(request.form['T'])

    values = [p1, p2, p3, p4, p5, p6, p7, p8]
    values_scaled = scaler.transform([values])
    prediction = model.predict(values_scaled)[0]

    return render_template('result.html',
                           prediction_text=f"Benzene Level: {prediction:.2f} µg/m³")
if __name__ == "__main__":
    app.run(debug=True, port=0)


