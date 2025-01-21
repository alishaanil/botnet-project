import numpy as np
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
from imblearn.over_sampling import SMOTE

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Load the dataset
data = pd.read_csv("cleaned_data1.csv")

# Preview the dataset
print(data.head())

# Separate features (X) and target (y)
X = data.drop(columns=['label'])  # Replace 'label' with the actual name of the target column
y = data['label']

# Check the original class distribution
print("Original class distribution:", Counter(y))

# Apply SMOTE for oversampling
try:
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_balanced, y_balanced = smote.fit_resample(X, y)
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

# confusion metrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=dt_model.classes_, yticklabels=dt_model.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Binarize the output (required for ROC-AUC for multi-class classification)
y_test_binarized = label_binarize(y_test, classes=dt_model.classes_)
n_classes = y_test_binarized.shape[1]

# Train a One-vs-Rest (OvR) classifier using the Decision Tree
ovr_classifier = OneVsRestClassifier(DecisionTreeClassifier(criterion='gini', random_state=42, class_weight='balanced'))
ovr_classifier.fit(X_train, label_binarize(y_train, classes=dt_model.classes_))

# Get the predicted probabilities for each class
y_score = ovr_classifier.predict_proba(X_test)

# Compute ROC curve and ROC-AUC for each class
fpr = {}
tpr = {}
roc_auc = {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot the ROC curves for each class
plt.figure(figsize=(10, 8))

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {dt_model.classes_[i]} (AUC = {roc_auc[i]:.2f})')

# Plot the diagonal (random classifier)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.50)')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Multi-Class Classification')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# If you want to calculate the macro-average AUC
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)

for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

macro_auc = auc(all_fpr, mean_tpr)
print("Macro-Average AUC:", macro_auc)

# corelation metrix
# Select only numeric columns
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Compute the correlation matrix
corr_matrix = numeric_data.corr()


plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()
