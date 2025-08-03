from flask import Flask, request, render_template, flash
from test import predict_fraud, _valid

app = Flask(__name__)
app.secret_key = 'your-secret-key'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            txn = {
                'time_since_login_min': float(request.form['time_since_login_min']),
                'transaction_amount':    float(request.form['transaction_amount']),
                'transaction_type':      request.form['transaction_type'],
                'is_first_transaction':  1 if request.form.get('is_first_transaction') else 0,
                'user_tenure_months':    float(request.form['user_tenure_months'])
            }
            out = predict_fraud(txn)
            sus = out['fraud_probability']
            if out['is_fraud']:
                flash(f"(sus level - {sus}%) fraudulent activity detected, please halt", 'danger')
            else:
                flash(f"(sus level - {sus}%) not a fraud, you may proceed", 'success')
        except ValueError as e:
            flash(str(e), 'warning')
    return render_template('index.html', valid_types=sorted(_valid))

if __name__ == '__main__':
    app.run(debug=True)