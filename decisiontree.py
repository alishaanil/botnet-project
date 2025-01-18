import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from collections import Counter
from sklearn.model_selection import cross_val_score

# Load the dataset
data = pd.read_csv("cleaned_data.csv")

# Preview the dataset
print(data.head())

# Separate features (X) and target (y)
X = data.drop(columns=['label'])  # Replace 'label' with the actual name of the target column
y = data['label']

# Check the original class distribution
print("Original class distribution:", Counter(y))

# Apply hybrid resampling (SMOTE + ENN)
try:
    smote_enn = SMOTEENN(smote=SMOTE(random_state=42, k_neighbors=2), random_state=42)
    X_balanced, y_balanced = smote_enn.fit_resample(X, y)
    print("Balanced class distribution:", Counter(y_balanced))
except ValueError as e:
    print(f"Resampling Error: {e}")
    X_balanced, y_balanced = X, y  # Fallback to original data

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Decision Tree model
# You can adjust hyperparameters like max_depth, min_samples_split, etc., for better performance
dt_model = DecisionTreeClassifier(criterion='gini', random_state=42, class_weight='balanced')

# Train the Decision Tree model
dt_model.fit(X_train, y_train)

# Make predictions
y_pred = dt_model.predict(X_test)

# Evaluate the Decision Tree model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print("Accuracy Score:", accuracy_score(y_test, y_pred))


# Perform cross-validation
cv_scores = cross_val_score(dt_model, X_train, y_train, cv=5, scoring='accuracy')  # cv=5 for 5-fold cross-validation

# Print CV scores
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())
print("Standard Deviation of CV Scores:", cv_scores.std())
