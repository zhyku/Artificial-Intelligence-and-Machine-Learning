import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load your CSV file
df = pd.read_csv('data.csv')  # Replace with your actual CSV file path

# Copy dataset
data = df.copy()

# Encode transaction_type
label_encoder = LabelEncoder()
data['transaction_type'] = label_encoder.fit_transform(data['transaction_type'])

# Drop timestamp
data.drop(columns=['timestamp'], inplace=True)

# Features and target
X = data.drop(columns=['is_fraud', 'predicted_fraud', 'predicted_fraud_proba'])
y = data['is_fraud']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)
feature_cols = X.columns.tolist()  # X = your features dataframe after preprocessing
joblib.dump(feature_cols, 'feature_cols.joblib')

# Predict on test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2%}")

# Save everything
joblib.dump(model, 'fraud_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')

print("Model retrained and saved successfully!")