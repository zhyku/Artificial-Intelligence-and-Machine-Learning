import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# 1. Load and preprocess
df = pd.read_csv('C:/Users/twaji/OneDrive/Desktop/RTRP/data/large_ratings.csv')
# Average duplicate ratings per user-item
df = df.groupby(['user_id', 'product_id'], as_index=False)['rating'].mean()

# 2. Filter sparse users/items
min_user_ratings = 5
min_item_ratings = 10
user_counts = df['user_id'].value_counts()
item_counts = df['product_id'].value_counts()
df = df[df['user_id'].isin(user_counts[user_counts >= min_user_ratings].index)]
df = df[df['product_id'].isin(item_counts[item_counts >= min_item_ratings].index)]

# 3. Train/test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 4. Helper functions
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

class NMFRecommender:
    def __init__(self, n_components=20, max_iter=200,
                 alpha=0.0, l1_ratio=0.0, init='nndsvda', random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.init = init
        self.random_state = random_state

    def fit(self, df_train):
        # Index maps
        self.user_ids = sorted(df_train.user_id.unique())
        self.item_ids = sorted(df_train.product_id.unique())
        self.u2i = {u: i for i, u in enumerate(self.user_ids)}
        self.i2i = {j: i for i, j in enumerate(self.item_ids)}
        # Build rating matrix
        M = df_train.pivot_table(
            index='user_id', columns='product_id', values='rating', fill_value=0
        ).reindex(index=self.user_ids, columns=self.item_ids, fill_value=0).values
        # Demean per user
        self.user_means = df_train.groupby('user_id')['rating'].mean().reindex(self.user_ids).values
        M_demeaned = M - self.user_means[:, np.newaxis]
        # Fit NMF model
        self.model = NMF(
            n_components=self.n_components,
            init=self.init,
            max_iter=self.max_iter,
            random_state=self.random_state,
            alpha_W=self.alpha,
            alpha_H=self.alpha,
            l1_ratio=self.l1_ratio
        )
        self.W = self.model.fit_transform(np.clip(M_demeaned, 0, None))
        self.H = self.model.components_
        return self

    def predict_rating(self, user_id, item_id):
        if user_id not in self.u2i or item_id not in self.i2i:
            return np.nan
        ui, ii = self.u2i[user_id], self.i2i[item_id]
        raw = self.W[ui].dot(self.H[:, ii]) + self.user_means[ui]
        return float(np.clip(raw, 1, 5))

    def evaluate(self, df_eval):
        preds = df_eval.apply(
            lambda row: self.predict_rating(row.user_id, row.product_id), axis=1
        )
        mask = ~preds.isna()
        return rmse(df_eval.rating[mask], preds[mask])

    def recommend_top_k(self, user_id, k=10, df_full=None):
        # df_full if provided to filter seen
        scores = []
        for item in self.item_ids:
            scores.append((item, self.predict_rating(user_id, item)))
        seen = set()
        if df_full is not None:
            seen = set(df_full[df_full.user_id == user_id].product_id)
        recs = [t for t in sorted(scores, key=lambda x: x[1], reverse=True) if t[0] not in seen]
        return recs[:k]

# 5. Hyperparameter grid search on hold-out split
grid = {
    'n_components': [10, 20, 30],
    'max_iter': [200, 500],
    'alpha': [0.0, 0.1],
    'l1_ratio': [0.0, 0.5]
}
best_score = np.inf
best_params = None
for k in grid['n_components']:
    for it in grid['max_iter']:
        for a in grid['alpha']:
            for l in grid['l1_ratio']:
                rec = NMFRecommender(n_components=k, max_iter=it, alpha=a, l1_ratio=l)
                rec.fit(train_df)
                score = rec.evaluate(test_df)
                print(f"k={k}, iters={it}, alpha={a}, l1={l} -> RMSE={score:.4f}")
                if score < best_score:
                    best_score = score
                    best_params = {'n_components': k, 'max_iter': it, 'alpha': a, 'l1_ratio': l}
print(f"\nBest params: {best_params} with RMSE={best_score:.4f}")

# 6. Train final model on full data
if best_params is None:
    print("No valid params found—using defaults.")
    best_params = {'n_components': 20, 'max_iter': 200, 'alpha': 0.0, 'l1_ratio': 0.0}
final_rec = NMFRecommender(**best_params).fit(df)

# 7. Example: top-10 recommendations for a user
user_sample = final_rec.user_ids[0]
print(f"Top-10 recommendations for user {user_sample}:",
      final_rec.recommend_top_k(user_sample, k=10, df_full=df))

# 8. Ranking metrics: Precision@K, Recall@K, NDCG@K
def precision_at_k(recs, actual, k):
    rec_set = [item for item, _ in recs[:k]]
    hits = len(set(rec_set) & set(actual))
    return hits / k

def recall_at_k(recs, actual, k):
    rec_set = [item for item, _ in recs[:k]]
    hits = len(set(rec_set) & set(actual))
    return hits / len(actual) if actual else 0.0

def dcg_at_k(recs, actual, k):
    dcg = 0.0
    for i, (item, _) in enumerate(recs[:k]):
        if item in actual:
            dcg += 1.0 / np.log2(i + 2)
    return dcg

def idcg_at_k(actual, k):
    # ideal DCG: hits sorted at top
    ideal_hits = min(len(actual), k)
    return sum((1.0 / np.log2(i + 2) for i in range(ideal_hits)))

def ndcg_at_k(recs, actual, k):
    idcg = idcg_at_k(actual, k)
    return dcg_at_k(recs, actual, k) / idcg if idcg > 0 else 0.0

# 9. Evaluate ranking metrics on test set
ks = [5, 10]
precisions, recalls, ndcgs = {k: [] for k in ks}, {k: [] for k in ks}, {k: [] for k in ks}
# build ground truth per user from test_df
test_true = test_df.groupby('user_id')['product_id'].apply(list).to_dict()
for user in final_rec.user_ids:
    recs = final_rec.recommend_top_k(user, k=max(ks), df_full=train_df)
    actual = test_true.get(user, [])
    if not actual:
        continue
    for k in ks:
        precisions[k].append(precision_at_k(recs, actual, k))
        recalls[k].append(recall_at_k(recs, actual, k))
        ndcgs[k].append(ndcg_at_k(recs, actual, k))
# print average metrics
for k in ks:
    print(f"Precision@{k}: {np.mean(precisions[k]):.4f}")
    print(f"Recall@{k}:    {np.mean(recalls[k]):.4f}")
    print(f"NDCG@{k}:     {np.mean(ndcgs[k]):.4f}")

# 10. Save model + artifacts
joblib.dump(final_rec.model, 'nmf_model.joblib')
joblib.dump(final_rec.W, 'user_factors.joblib')
joblib.dump(final_rec.H, 'item_factors.joblib')
joblib.dump(final_rec.u2i, 'user_index_map.joblib')
joblib.dump(final_rec.i2i, 'item_index_map.joblib')
joblib.dump(final_rec.user_means, 'user_means.joblib')
print(f"Saved final model and artifacts with params {best_params}.")
