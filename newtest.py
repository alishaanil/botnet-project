import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


# Example: Load your dataset (replace with your actual dataset)
df = pd.read_csv('cleaned_data.csv')



# Column to exclude from scaling
exclude_column = 'label'

# Initialize the scaler
scaler = MinMaxScaler()

# Apply scaling to all columns except the one to exclude
columns_to_scale = df.columns.difference([exclude_column])
scaled_data = scaler.fit_transform(df[columns_to_scale])

# Convert the scaled data back to a DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=columns_to_scale)

# Add the excluded column back to the DataFrame
scaled_df[exclude_column] = df[exclude_column].reset_index(drop=True)

#print("Original DataFrame:\n", df)
#print("\nScaled DataFrame (with one column excluded):\n", scaled_df)

numeric_columns = df.select_dtypes(include=['float64', 'int64'])

# Calculate the correlation matrix for numeric columns only
correlation_matrix = numeric_columns.corr()

# Plot the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()
