from flask import Flask, request, render_template, flash, redirect, url_for
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
                flash(f"(sus level - {sus}%) Fraudulent Activity Detected, Please Halt!", 'danger')
            else:
                flash(f"(sus level - {sus}%) No Fraud Detected, You May Proceed...", 'success')
        except ValueError as e:
            flash(str(e), 'warning')
        return redirect(url_for('index'))  # <--- This makes the flash show once and clears on refresh
    return render_template('index.html', valid_types=sorted(_valid))


if __name__ == '__main__':
    app.run(debug=True)