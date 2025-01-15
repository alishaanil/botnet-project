import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load your dataset
data = pd.read_csv('cleaned_data.csv')

# Step 2: Check the target column (Replace 'target_column' with the actual name of your target column)
target_column = 'label'  # Replace with the actual name of your target column
class_counts = data[target_column].value_counts()

# Step 3: Print the class distribution
print(class_counts)

# Step 4: Plot the class distribution
class_counts.plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()
