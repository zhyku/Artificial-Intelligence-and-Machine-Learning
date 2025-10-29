from flask import Flask, request, render_template, flash, redirect, url_for, jsonify
from test import predict_fraud, _valid

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Change this for production!

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            txn = {
                'time_since_login_min': float(request.form['time_since_login_min']),
                'transaction_amount': float(request.form['transaction_amount']),
                'transaction_type': request.form['transaction_type'],
                'is_first_transaction': int(request.form['is_first_transaction']),
                'user_tenure_months': float(request.form['user_tenure_months'])
            }
            out = predict_fraud(txn)
            sus = round(out['fraud_probability'] * 100, 2)
            if out['is_fraud']:
                flash(f"(sus level - {sus}%) Fraudulent Activity Detected, Please Halt!", 'danger')
            else:
                flash(f"(sus level - {sus}%) No Fraud Detected, You May Proceed...", 'success')
        except Exception as e:
            flash(f"Input error: {e}", 'warning')
        return redirect(url_for('index'))
    return render_template(
        'index.html',
        valid_types=[v.replace('transaction_type_', '') for v in sorted(_valid)]
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        txn = {
            'time_since_login_min': float(data['time_since_login_min']),
            'transaction_amount': float(data['transaction_amount']),
            'transaction_type': data['transaction_type'],
            'is_first_transaction': int(data['is_first_transaction']),
            'user_tenure_months': float(data['user_tenure_months'])
        }
        out = predict_fraud(txn)
        return jsonify({
            'is_fraud': bool(out['is_fraud']),
            'fraud_probability': float(out['fraud_probability'])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
