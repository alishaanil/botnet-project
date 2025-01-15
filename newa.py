import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('cleaned_data.csv')

# Basic structure
print(data.info())
print(data.describe())
print(data['label'].value_counts())
print(data.columns)

# Class distribution
data['label'].value_counts().plot(kind='bar', color=['blue', 'red'])
plt.title('Class Distribution: Benign vs Malicious')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()
