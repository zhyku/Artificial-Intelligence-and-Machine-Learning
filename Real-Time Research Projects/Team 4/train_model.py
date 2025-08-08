import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set pandas display format for floats
pd.options.display.float_format = "{:.2f}".format

# Define and verify data path
data_path = "data/bike_data.csv"  # Relative path
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, data_path)

# Check if file exists
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at: {data_path}. Please ensure 'bike_data.csv' is in the 'data/' directory.")

# Load dataset
df = pd.read_csv(data_path, encoding="ISO-8859-1")

# Preprocess data
def preprocess_data(data):
    # Create a copy to avoid modifying the original
    data = data.copy()
    
    # Initialize LabelEncoders for categorical columns
    le_seasons = LabelEncoder()
    le_holiday = LabelEncoder()
    le_functioning = LabelEncoder()
    
    # Encode categorical columns
    data['Seasons'] = le_seasons.fit_transform(data['Seasons'])
    data['Holiday'] = le_holiday.fit_transform(data['Holiday'])
    data['Functioning Day'] = le_functioning.fit_transform(data['Functioning Day'])
    
    # Drop 'Date' as it's not used directly in the model
    data = data.drop(['Date'], axis=1)
    
    # Save LabelEncoders
    joblib.dump(le_seasons, 'model/le_seasons.pkl')
    joblib.dump(le_holiday, 'model/le_holiday.pkl')
    joblib.dump(le_functioning, 'model/le_functioning.pkl')
    
    return data, le_seasons, le_holiday, le_functioning

# Prepare features and target
df_processed, le_seasons, le_holiday, le_functioning = preprocess_data(df)
X = df_processed.drop('Rented Bike Count', axis=1)
y = df_processed['Rented Bike Count']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
numerical_cols = ['Hour', 'Temperature(C)', 'Humidity(%)', 'Wind speed (m/s)', 
                 'Visibility (10m)', 'Dew point temperature(C)', 'Solar Radiation (MJ/m2)', 
                 'Rainfall(mm)', 'Snowfall (cm)']
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Save scaler
joblib.dump(scaler, 'model/scaler.pkl')

# Define hyperparameter grid for RandomForestRegressor
param_grid = {
    'n_estimators': [50, 80, 100],
    'max_depth': [4, 6, 8],
    'min_samples_split': [50, 100, 150],
    'min_samples_leaf': [40, 50]
}

# Initialize RandomForestRegressor and GridSearchCV
rf = RandomForestRegressor()
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    verbose=2
)

# Fit the model
grid_search.fit(X_train, y_train)

# Save the trained model
joblib.dump(grid_search.best_estimator_, 'model/random_forest_model.pkl')

# Print best parameters and score
print(f"\nBest R2 Score: {grid_search.best_score_:.6f}")
print(f"Best Parameters: {grid_search.best_params_}")

# Make predictions
y_pred_train = grid_search.predict(X_train)
y_pred_test = grid_search.predict(X_test)

# Calculate metrics
def calculate_metrics(y_true, y_pred, X_data, dataset_name):
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y_true) - 1) / (len(y_true) - X_data.shape[1] - 1)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    return {
        'Dataset': dataset_name,
        'R2': r2,
        'Adjusted R2': adj_r2,
        'MAE': mae,
        'RMSE': rmse
    }

# Store and display results
results = pd.DataFrame([
    calculate_metrics(y_train, y_pred_train, X_train, 'Train'),
    calculate_metrics(y_test, y_pred_test, X_test, 'Test')
])
print("\nModel Performance Metrics:")
print(results)

# Save results to CSV
results.to_csv('model/model_results.csv', index=False)

print("\nModel and preprocessors saved to 'model/' directory.")