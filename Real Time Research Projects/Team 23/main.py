import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Load dataset
df = pd.read_csv("Sample - Superstore.csv", encoding='latin1')
df['Order Date'] = pd.to_datetime(df['Order Date'])

def get_categories_and_subcategories():
    """
    Return a dictionary mapping categories to their sub-categories.
    """
    category_map = {}
    for category in df['Category'].unique():
        subcats = df[df['Category'] == category]['Sub-Category'].unique().tolist()
        category_map[category] = subcats
    return category_map

def prepare_data(category=None, sub_category=None):
    """
    Filter data by category/sub-category and aggregate monthly sales.
    """
    filtered_df = df.copy()

    if category:
        filtered_df = filtered_df[filtered_df['Category'] == category]
    if sub_category:
        filtered_df = filtered_df[filtered_df['Sub-Category'] == sub_category]

    if filtered_df.empty:
        raise ValueError("No data available for selected Category and Sub-Category.")

    monthly_sales = (
        filtered_df
        .groupby(pd.Grouper(key='Order Date', freq='MS'))['Sales']
        .sum()
        .reset_index()
    )
    monthly_sales.set_index('Order Date', inplace=True)
    return monthly_sales

def train_arima_model(data):
    """
    Train an ARIMA(1,1,1) model.
    """
    model = ARIMA(data, order=(1, 1, 1))
    model_fit = model.fit()
    return model_fit

def predict_sales(future_date, category=None, sub_category=None, current_price=None):
    """
    Predict future sales for the given filters and future date.
    If current_price is given, adjust the forecast using it.
    """
    future_date = pd.to_datetime(future_date)
    sales_data = prepare_data(category, sub_category)
    model = train_arima_model(sales_data)

    last_date = sales_data.index[-1]
    steps_ahead = (future_date.year - last_date.year) * 12 + (future_date.month - last_date.month)

    if steps_ahead <= 0:
        raise ValueError("Please select a future date after the last available date in the dataset.")

    forecast = model.forecast(steps=steps_ahead)
    forecasted_value = forecast.iloc[-1]

    # Adjust forecast using current_price if provided and valid
    if current_price is not None and current_price != "":
        try:
            current_price = float(current_price)
            past_mean = sales_data['Sales'][-6:].mean()  # last 6 months mean
            if past_mean != 0 and not np.isnan(past_mean):
                adjustment_ratio = current_price / past_mean
                forecasted_value *= adjustment_ratio
        except ValueError:
            # In case conversion to float fails
            pass

    return forecasted_value
