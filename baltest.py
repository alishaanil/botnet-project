# Install the required libraries:
# pip install imbalanced-learn scikit-learn

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from collections import Counter




import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Step 1: Load your dataset
data = pd.read_csv('cleaned_data.csv')

# Step 2: Split the dataset into features (X) and target (y)
X = data.drop('label', axis=1)  # Replace 'target_column' with the actual target column name
y = data['label']  # Replace 'target_column' with the actual target column name

# Step 3: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Initialize SMOTE and apply it to the training data
smote = SMOTE(k_neighbors=1)  # You can set k_neighbors to a suitable number
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Now you can proceed with the resampled data (X_resampled, y_resampled) for training your model


# Step 1: Load your dataset
# Example: Assuming you have a CSV file
#data = pd.read_csv('cleaned_data.csv')

#from imblearn.over_sampling import SMOTE

# Initialize SMOTE with fewer neighbors
#smote = SMOTE(k_neighbors=1)  # or 2 if there are at least 2 samples in the minority class

# Resample the data
#X_resampled, y_resampled = smote.fit_resample(X_train, y_train)


# Separate features (X) and target variable (y)
X = data.drop('label', axis=1)  # Replace 'target_column' with the actual name of your target column
y = data['label']

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Apply SMOTE to balance the training set
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Step 4: Check the class distribution before and after SMOTE
print("Original dataset shape:", Counter(y_train))
print("Resampled dataset shape:", Counter(y_resampled))

# Step 5: Train a RandomForestClassifier on the resampled dataset
clf = RandomForestClassifier(random_state=42)
clf.fit(X_resampled, y_resampled)

# Step 6: Evaluate the model on the test set
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
