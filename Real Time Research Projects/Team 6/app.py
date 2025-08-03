from flask import Flask, render_template, request, abort
import pandas as pd
import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD

app = Flask(__name__)

# --- Load data & models on startup ---
# Load main data
df = pd.read_csv("clean_data.csv")

# Load trending products and ensure 'ProdID' column exists
trending_df = pd.read_csv("trending_products.csv")
trending_df.rename(columns={trending_df.columns[0]: 'ProdID'}, inplace=True)
trending = trending_df['ProdID'].tolist()

# Load SVD model
svd = joblib.load("svd_model.joblib")

# Build user-item matrix & prediction matrix
rating_matrix = df.pivot_table(
    index='ID', columns='ProdID', values='Rating'
).fillna(0)

# Precompute full prediction matrix
full_pred = svd.inverse_transform(svd.transform(rating_matrix.values))

pred_df = pd.DataFrame(
    full_pred,
    index=rating_matrix.index,
    columns=rating_matrix.columns
)

# Helper: get top-N similar items for a given ProdID
def get_item_recs(prodid, n=5):
    if prodid not in pred_df.columns:
        return []  # Return empty if product not found
    item_vec = pred_df[prodid].values.reshape(1, -1)
    all_vecs = pred_df.values.T  # shape (items, users)
    sims = np.dot(all_vecs, item_vec.T).flatten()
    idxs = np.argsort(sims)[::-1]
    prod_ids = pred_df.columns[idxs]
    # skip the prodid itself
    prod_ids = [p for p in prod_ids if p != prodid]
    return prod_ids[:n]

# --- Routes ---

@app.route("/")
def index():
    # Show trending and all products
    products = df.drop_duplicates('ProdID')[['ProdID', 'Name', 'ImageURL']]
    trending_products = products[products['ProdID'].isin(trending)]
    return render_template(
        "index.html",
        products=products,
        trending=trending_products
    )

@app.route("/recommend/<prodid>")
def recommend(prodid):
    # Validate product ID
    products_unique = df.drop_duplicates(subset='ProdID').set_index('ProdID')
    if prodid not in products_unique.index:
        abort(404, description="Product not found")
    # 1. Get top 5 item-to-item recommendations
    rec_ids = get_item_recs(prodid, n=5)
    # 2. Assemble the recommendation DataFrame
    recs = products_unique.loc[rec_ids]
    # 3. Fetch the selected product’s own info
    product = products_unique.loc[prodid]
    # 4. Render the template
    return render_template(
        "recommend.html",
        product=product,
        recommendations=recs.reset_index()
    )

if __name__ == "__main__":
    app.run(debug=True)