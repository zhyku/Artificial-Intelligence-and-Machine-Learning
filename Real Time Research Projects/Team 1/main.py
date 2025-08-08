import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# 1. Load data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.drop('customerID', axis=1, inplace=True)

# 2. Clean data
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df["Churn"] = df["Churn"].map({'Yes': 1, 'No': 0})

# 3. Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# 4. Split into features and target
X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Handle imbalance with SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

# 8. XGBoost model tuning
xgb = XGBClassifier(random_state=42, eval_metric='logloss')
params = {
    'n_estimators': [200],
    'max_depth': [7],
    'learning_rate': [0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}
grid = GridSearchCV(xgb, params, cv=3, scoring='f1', verbose=1, n_jobs=-1)
grid.fit(X_train_res, y_train_res)

best_model = grid.best_estimator_
print("Best Parameters:", grid.best_params_)

# 9. Prediction with custom threshold
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
threshold = 0.4
y_pred = (y_proba >= threshold).astype(int)

# 10. Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 11. Save model, scaler, features, and threshold
joblib.dump(best_model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")
joblib.dump(threshold, "threshold.pkl")
joblib.dump(0.4, "threshold.pkl")

print("\nModel, scaler, features, and threshold saved.")
