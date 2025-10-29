import joblib
import pandas as pd

# Threshold for flagging fraud (70%)
THRESHOLD = 0.70

# Load saved objects
model = joblib.load('fraud_model.joblib')
scaler = joblib.load('scaler.joblib')
label_encoder = joblib.load('label_encoder.joblib')
feature_cols = [
    'time_since_login_min',
    'transaction_amount',
    'transaction_type',
    'is_first_transaction',
    'user_tenure_months'
]

# Valid transaction types (uppercased)
valid_types = {str(t).upper() for t in label_encoder.classes_ if pd.notna(t)}


def predict_fraud(transaction_dict):
    # Normalize & validate
    txn_type = str(transaction_dict['transaction_type']).upper()
    if txn_type not in valid_types:
        raise ValueError(f"Unknown transaction type: {transaction_dict['transaction_type']}")
    # Encode
    for orig in label_encoder.classes_:
        if pd.notna(orig) and orig.upper() == txn_type:
            encoded = label_encoder.transform([orig])[0]
            break

    # Build DataFrame
    df = pd.DataFrame([{
        'time_since_login_min': transaction_dict['time_since_login_min'],
        'transaction_amount':      transaction_dict['transaction_amount'],
        'transaction_type':        encoded,
        'is_first_transaction':    transaction_dict['is_first_transaction'],
        'user_tenure_months':      transaction_dict['user_tenure_months']
    }], columns=feature_cols)

    # Predict
    proba = model.predict_proba(scaler.transform(df))[0][1]
    return {
        'is_fraud': proba >= THRESHOLD,
        'fraud_probability': round(proba * 100, 2)
    }

# Expose for Flask
_valid = valid_types