import pandas as pd

# Step 1: Load your dataset
data = pd.read_csv('cleaned_data.csv')

# Step 2: Print the names of all columns
print("Column names in the dataset:")
print(data.columns)
