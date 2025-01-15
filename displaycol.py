import pandas as pd

# Set Pandas option to display all columns
pd.set_option('display.max_columns', None)

# Example: Load your dataset (replace with your actual dataset)
df = pd.read_csv('cleaned_data.csv')

# Display the DataFrame with all columns visible
print(df)
