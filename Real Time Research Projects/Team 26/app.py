from flask import Flask, render_template, request
import main  # Make sure this has predict_sales and prepare_data
import pandas as pd

app = Flask(__name__)

# Load data to extract category and sub-category options
df = pd.read_csv("Sample - Superstore.csv", encoding='latin1')
df.dropna(subset=['Category', 'Sub-Category'], inplace=True)

categories = sorted(df['Category'].unique())
sub_categories_by_category = {
    category: sorted(df[df['Category'] == category]['Sub-Category'].unique())
    for category in categories
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    selected_category = None
    selected_sub_category = None
    current_price = ''

    if request.method == 'POST':
        try:
            future_date = request.form['future_date']
            selected_category = request.form.get('category')
            selected_sub_category = request.form.get('sub_category')
            current_price_input = request.form.get('current_price')

            current_price = current_price_input.strip() if current_price_input else ''

            prediction = main.predict_sales(
                future_date=future_date,
                category=selected_category,
                sub_category=selected_sub_category,
                current_price=current_price
            )
        except Exception as e:
            error = f"An error occurred: {str(e)}"

    return render_template(
        'index.html',
        prediction=prediction,
        error=error,
        categories=categories,
        sub_categories=sub_categories_by_category,
        selected_category=selected_category,
        selected_sub_category=selected_sub_category,
        current_price=current_price
    )

if __name__ == '__main__':
    app.run(debug=True)
