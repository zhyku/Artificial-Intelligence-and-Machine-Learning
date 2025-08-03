from flask import Flask, request, render_template
from test import predict_fraud, _valid  # Make sure these are defined in your test.py

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Change this for production!

@app.route('/', methods=['GET', 'POST'])
def index():
    form_data = {}
    result = None
    result_category = None  # For coloring (danger/success/warning)
    if request.method == 'POST':
        try:
            form_data = request.form.to_dict()
            txn = {
                'time_since_login_min': float(request.form['time_since_login_min']),
                'transaction_amount': float(request.form['transaction_amount']),
                'transaction_type': request.form['transaction_type'],
                'is_first_transaction': 1 if request.form.get('is_first_transaction') else 0,
                'user_tenure_months': float(request.form['user_tenure_months'])
            }
            out = predict_fraud(txn)
            sus = round(out['fraud_probability'] * 100, 2)
            if out['is_fraud']:
                result = f"(sus level - {sus}%) Fraudulent Activity Detected, Please Halt!"
                result_category = "danger"
            else:
                result = f"(sus level - {sus}%) No Fraud Detected, You May Proceed..."
                result_category = "success"
        except Exception as e:
            result = f"Input error: {e}"
            result_category = "warning"
    return render_template(
        'index.html',
        valid_types=sorted(_valid),
        form_data=form_data,
        result=result,
        result_category=result_category
    )

if __name__ == '__main__':
    app.run(debug=True)
