from flask import Flask, send_from_directory, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__, static_folder='static')

# Load artifacts
data_df = pd.read_csv('data/large_ratings.csv')
item_popularity = data_df.groupby('product_id')['rating'].mean().sort_values(ascending=False).index.tolist()
# Example: load NMF model and maps if you want
# model = joblib.load('models/nmf_model.joblib')
# u2i = joblib.load('models/user_index_map.joblib')
# i2u = joblib.load('models/item_index_map.joblib')

@app.route('/')
def root():
    return send_from_directory('static', 'index.html')

@app.route('/api/random-items')
def random_items():
    # Return 5 random products
    items = data_df['product_id'].drop_duplicates().sample(5).tolist()
    return jsonify({'items': items})

@app.route('/api/recommend', methods=['POST'])
def recommend():
    ratings = request.json.get('ratings', {})
    exclude = set(map(int, ratings.keys()))
    recs = [int(pid) for pid in item_popularity if int(pid) not in exclude][:10]
    return jsonify({'recommendations': recs})

if __name__ == '__main__':
    app.run(debug=True)
