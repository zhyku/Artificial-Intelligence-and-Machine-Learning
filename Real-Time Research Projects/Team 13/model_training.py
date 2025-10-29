import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# 1. Load dataset
try:
   data = pd.read_csv(r"C:\Users\Rakesh\Desktop\RTRP\cleaned_data - Copy.csv")
except FileNotFoundError as e:
    print(f"Error loading dataset: {e}")
    raise

# 2. Drop timestamp column and any other non-numeric columns
data = data.drop(columns=['timestamp'], errors='ignore')

# 3. Check for missing values
if data.isnull().sum().any():
    print("Missing values detected. Filling with forward fill.")
    data.fillna(method='ffill', inplace=True)

# 4. Convert categorical variables using one-hot encoding
data = pd.get_dummies(data, drop_first=True)

# 5. Check for correlation and drop highly correlated features
correlation_matrix = data.corr()
threshold = 0.9
to_drop = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            to_drop.add(colname)
data = data.drop(columns=to_drop)

# 6. Remove target leakage columns from features
leakage_cols = ['predicted_fraud_proba', 'is_fraud']
X = data.drop(columns=leakage_cols, errors='ignore')
y = data['is_fraud']

# 7. Print feature names for confirmation
print("\nFeature names used in the model:")
print(X.columns.tolist())

# 8. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1, stratify=y)

# 9. Apply SMOTE to handle class imbalance
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print("\nAfter SMOTE oversampling:")
print(pd.Series(y_train_res).value_counts())

# 10. Scale features
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# 11. Save the scaler
joblib.dump(scaler, 'scaler.joblib')
print("Scaler saved as scaler.joblib")

# 12. GridSearchCV for XGBoost hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=1
)

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='f1',
    cv=3,
    verbose=2,
    n_jobs=-1
)

print("\nStarting GridSearchCV for XGBoost hyperparameter tuning...")
grid_search.fit(X_train_res, y_train_res)

print("Best parameters:", grid_search.best_params_)
print("Best F1-score (CV):", grid_search.best_score_)

# 13. Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_test_pred = best_model.predict(X_test)
print("\nTest set evaluation with best model:")
print("Classification Report:\n", classification_report(y_test, y_test_pred, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

# 14. Threshold optimization (optional)
y_test_proba = best_model.predict_proba(X_test)[:, 1]
THRESHOLD = 0.7
y_test_pred_thresh = (y_test_proba > THRESHOLD).astype(int)

print(f"\nEvaluation with threshold = {THRESHOLD}")
print("Classification Report:\n", classification_report(y_test, y_test_pred_thresh, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred_thresh))

# 15. Save the tuned model and feature names
joblib.dump(best_model, 'xgb_model_tuned.joblib')
print("Tuned model saved as xgb_model_tuned.joblib")
joblib.dump(X.columns.tolist(), 'feature_names.joblib')
print("Feature names saved as feature_names.joblib")
