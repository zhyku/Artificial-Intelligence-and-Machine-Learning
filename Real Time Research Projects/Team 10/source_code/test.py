import joblib
import pandas as pd

# Load the tuned model, scaler, and feature names
model = joblib.load('xgb_model_tuned.joblib')
scaler = joblib.load('scaler.joblib')
feature_names = joblib.load('feature_names.joblib')

# Lower base threshold and add dynamic threshold based on amount
BASE_THRESHOLD = 0.85
MAX_SAFE_AMOUNT = 1000000  # Amount above which we should be more strict

# Valid transaction types
_valid = ['transaction_type_PAYMENT', 'transaction_type_TRANSFER', 'transaction_type_CASH_OUT']

def predict_fraud(transaction):
    # Create a sample with all features set to 0
    sample = {name: 0 for name in feature_names}
    
    # Update with the provided transaction details
    transaction_type = f"transaction_type_{transaction['transaction_type']}"
    if transaction_type in _valid:
        sample[transaction_type] = 1
    
    sample.update({
        'time_since_login_min': transaction['time_since_login_min'],
        'transaction_amount': transaction['transaction_amount'],
        'is_first_transaction': transaction['is_first_transaction'],
        'user_tenure_months': transaction['user_tenure_months']
    })
    
    # Create DataFrame and ensure correct column order
    df = pd.DataFrame([sample])
    df = df[feature_names]
    
    # Scale features and predict
    scaled = scaler.transform(df)
    proba = model.predict_proba(scaled)[0][1]
    
    # Dynamic threshold based on amount and tenure
    threshold = BASE_THRESHOLD
    
    # Lower threshold for high amounts
    if transaction['transaction_amount'] > MAX_SAFE_AMOUNT:
        threshold *= 0.8  # 20% lower threshold for high amounts
    
    # Lower threshold for new users (less than 6 months tenure)
    if transaction['user_tenure_months'] < 6:
        threshold *= 0.8  # 20% lower threshold for new users
    
    # Lower threshold for first transactions
    if transaction['is_first_transaction']:
        threshold *= 0.8  # 20% lower threshold for first transactions
    
    is_fraud = proba > threshold
    
    return {
        'is_fraud': is_fraud,
        'fraud_probability': proba
    }

# Test samples for demonstration
if __name__ == '__main__':
    # Example 1: Very safe transaction (should be NOT FRAUD)
    sample_safe = {
        'time_since_login_min': 120,
        'transaction_amount': 10,
        'is_first_transaction': 0,
        'user_tenure_months': 48,
        'transaction_type': 'PAYMENT'
    }

    # Example 2: Risky transaction (should be FRAUD)
    sample_fraud = {
        'time_since_login_min': 0,
        'transaction_amount': 10000000,
        'is_first_transaction': 1,
        'user_tenure_months': 0,
        'transaction_type': 'TRANSFER'
    }

    # Test predictions
    samples = [
        ("Very safe transaction", sample_safe),
        ("Risky transaction", sample_fraud),
    ]

    for desc, sample in samples:
        result = predict_fraud(sample)
        pred_label = 'FRAUD' if result['is_fraud'] else 'NOT FRAUD'
        print(f"{desc}: {pred_label} (probability of fraud: {result['fraud_probability']:.4f})")

