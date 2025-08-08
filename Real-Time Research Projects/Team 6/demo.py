import pandas as pd

# Read the CSV as usual
trending_df = pd.read_csv("trending_products.csv")

# Rename the first column to 'ProdID' (if it's not already)
trending_df.rename(columns={trending_df.columns[0]: 'ProdID'}, inplace=True)

# Now you can safely use:
trending = trending_df['ProdID'].tolist()
