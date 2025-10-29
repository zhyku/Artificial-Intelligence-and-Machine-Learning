from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Correct model directory path
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')

# Debug prints
print("Model directory:", model_dir)
if os.path.exists(model_dir):
    print("Files in model directory:", os.listdir(model_dir))
else:
    print("Model directory does not exist!")

# Load model & encoders with error handling
try:
    model = joblib.load(os.path.join(model_dir, 'random_forest_model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    le_seasons = joblib.load(os.path.join(model_dir, 'le_seasons.pkl'))
    le_holiday = joblib.load(os.path.join(model_dir, 'le_holiday.pkl'))
    le_functioning = joblib.load(os.path.join(model_dir, 'le_functioning.pkl'))
except Exception as e:
    print(f"Error loading model or encoders: {e}")
    model = scaler = le_seasons = le_holiday = le_functioning = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if None in (model, scaler, le_seasons, le_holiday, le_functioning):
        return jsonify({'error': 'Model or encoders not loaded.'}), 500

    try:
        data = request.get_json(force=True)
        required_keys = [
            'hour', 'temperature', 'humidity', 'wind_speed', 'visibility',
            'dew_point', 'solar_radiation', 'rainfall', 'snowfall',
            'seasons', 'holiday', 'functioning_day'
        ]
        if not all(key in data for key in required_keys):
            return jsonify({'error': 'Missing required input fields.'}), 400

        input_df = pd.DataFrame([{
            'Hour': data['hour'],
            'Temperature(C)': data['temperature'],
            'Humidity(%)': data['humidity'],
            'Wind speed (m/s)': data['wind_speed'],
            'Visibility (10m)': data['visibility'],
            'Dew point temperature(C)': data['dew_point'],
            'Solar Radiation (MJ/m2)': data['solar_radiation'],
            'Rainfall(mm)': data['rainfall'],
            'Snowfall (cm)': data['snowfall'],
            'Seasons': le_seasons.transform([data['seasons']])[0],
            'Holiday': le_holiday.transform([data['holiday']])[0],
            'Functioning Day': le_functioning.transform([data['functioning_day']])[0]
        }])

        numerical_cols = [
            'Hour', 'Temperature(C)', 'Humidity(%)', 'Wind speed (m/s)',
            'Visibility (10m)', 'Dew point temperature(C)', 'Solar Radiation (MJ/m2)',
            'Rainfall(mm)', 'Snowfall (cm)'
        ]
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
        prediction = model.predict(input_df)[0]
        return jsonify({'prediction': round(float(prediction), 0)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
