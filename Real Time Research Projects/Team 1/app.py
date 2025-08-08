from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and preprocessing objects
model = joblib.load("model/churn_model.pkl")
features = joblib.load("model/features.pkl")
scaler = joblib.load("model/scaler.pkl")
threshold = joblib.load("model/threshold.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        form = request.form

        # Extract input values
        input_data = {
            'gender': form['gender'],
            'SeniorCitizen': form['SeniorCitizen'],
            'Partner': form['Partner'],
            'PhoneService': form['PhoneService'],
            'tenure': float(form['tenure']),
            'MonthlyCharges': float(form['MonthlyCharges']),
            'TotalCharges': float(form['TotalCharges']),
            'InternetService': form['InternetService'],
            'Contract': form['Contract']
        }

        # Create DataFrame and encode
        df = pd.DataFrame([input_data])
        df_encoded = pd.get_dummies(df).reindex(columns=features, fill_value=0)

        # Scale
        df_scaled = scaler.transform(df_encoded)

        # Predict
        pred_proba = model.predict_proba(df_scaled)[0][1]
        prediction = "Churn" if pred_proba > threshold else "No Churn"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
