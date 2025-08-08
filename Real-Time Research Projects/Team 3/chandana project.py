from flask import Flask, request, jsonify, render_template_string
import numpy as np

app = Flask(__name__)

# Linear Regression Coefficients (from training)
COEFS = [0.2593354, -6.51203535, -208.85870888]
INTERCEPT = 29.475640732952964

# HTML Template
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Profit Predictor</title>
  <style>
    body { font-family: Arial; margin: 40px; }
    input { margin: 10px; padding: 5px; width: 200px; }
    button { padding: 10px; }
  </style>
</head>
<body>
  <h2>Predict Profit</h2>
  <form id="predict-form">
    <label>Sales: <input type="number" step="0.01" name="sales" required></label><br>
    <label>Quantity: <input type="number" name="quantity" required></label><br>
    <label>Discount: <input type="number" step="0.01" name="discount" required></label><br>
    <button type="submit">Predict</button>
  </form>
  <h3 id="result"></h3>

  <script>
    const form = document.getElementById('predict-form');
    form.onsubmit = async (e) => {
      e.preventDefault();
      const formData = new FormData(form);
      const data = Object.fromEntries(formData);
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
      const result = await response.json();
      document.getElementById('result').innerText = Predicted Profit: $${result.profit.toFixed(2)};
    };
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        sales = float(data['sales'])
        quantity = float(data['quantity'])
        discount = float(data['discount'])
        features = np.array([sales, quantity, discount])
        profit = np.dot(COEFS, features) + INTERCEPT
        return jsonify({'profit': profit})
    except (KeyError, ValueError):
        return jsonify({'error': 'Invalid input'}), 400

if __name__ == '_main_':
    app.run(debug=True)