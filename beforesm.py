import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


df = pd.read_csv('cleaned_data.csv')



# Separate the features and target variable
X = df.drop(columns=['label'])  # Replace 'target' with your actual target column
y = df['label']  # Target variable


# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check class distribution in training set before SMOTE
print("Class distribution in training set before SMOTE:")
print(y_train.value_counts())

print("Min values before scaling:\n", X_train.min())
print("\nMax values before scaling:\n", X_train.max())



scaled_columns = columns_to_scale

# Creating a DataFrame from the scaled training data (after SMOTE)
X_resampled_scaled_df = pd.DataFrame(X_resampled_scaled, columns=X_train_res.columns)

# Check the minimum and maximum values of the scaled columns
print("Min values of scaled columns:\n", X_resampled_scaled_df[scaled_columns].min())
print("\nMax values of scaled columns:\n", X_resampled_scaled_df[scaled_columns].max())

# You can also check the scaling for the test set
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("\nMin values of scaled columns (Test set):\n", X_test_scaled_df[scaled_columns].min())
print("\nMax values of scaled columns (Test set):\n", X_test_scaled_df[scaled_columns].max())

conf_matrix = confusion_matrix(y_test, y_pred)

# Display confusion matrix as a heatmap
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
