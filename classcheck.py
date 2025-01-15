import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (replace 'your_dataset.csv' with your actual dataset)
df = pd.read_csv('cleaned_data.csv')


# Example: Assume 'label' is the target column in your dataset
target_column = 'label'  # Replace with the actual target column name

# Check the distribution of the target variable
class_distribution = df[target_column].value_counts()

print("\nClass Distribution:\n", class_distribution)

# Plot the class distribution
plt.figure(figsize=(8, 6))
sns.barplot(x=class_distribution.index, y=class_distribution.values)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()
