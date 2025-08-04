# train_cf_model.py

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

def load_data(path):
    """Load cleaned e-commerce data into a DataFrame."""
    df = pd.read_csv(path)
    return df

def build_rating_matrix(df):
    """Pivot to a user–item matrix, filling missing ratings with zero."""
    rating_matrix = df.pivot_table(
        index='ID',       # user ID
        columns='ProdID', # product ID
        values='Rating'
    ).fillna(0)
    return rating_matrix

def train_test_matrices(rating_matrix, test_size=0.2, random_state=42):
    """Split the user–item matrix into train and test portions by users."""
    train_mat, test_mat = train_test_split(
        rating_matrix, test_size=test_size, random_state=random_state
    )
    return train_mat.values, test_mat.values

def train_svd(train_array, n_components=20, random_state=42):
    """Fit TruncatedSVD on the training user–item matrix."""
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    svd.fit(train_array)
    return svd

def evaluate_model(svd, test_array):
    """
    Reconstruct test ratings from the SVD factors and compute RMSE
    only on the entries that are non-zero in the test matrix.
    """
    reconstructed = svd.inverse_transform(svd.transform(test_array))
    mask = test_array != 0
    mse = mean_squared_error(test_array[mask], reconstructed[mask])
    rmse = np.sqrt(mse)
    return rmse

def main():
    # 1. Load
    df = load_data("clean_data.csv")

    # 2. Build rating matrix
    rating_matrix = build_rating_matrix(df)

    # 3. Split
    train_array, test_array = train_test_matrices(rating_matrix)

    # 4. Train
    svd = train_svd(train_array, n_components=20)

    # 5. Evaluate
    rmse = evaluate_model(svd, test_array)
    print(f"Test RMSE (TruncatedSVD): {rmse:.4f}")

    # 6. Save model
    model_path = "svd_model.joblib"
    joblib.dump(svd, model_path)
    print(f"✅ Saved trained SVD model to '{model_path}'")

if __name__ == "__main__":
    main()
