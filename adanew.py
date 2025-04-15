import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Load the dataset
df = pd.read_csv('combined_file_3.csv', low_memory=False)

# Manually specify the features to include
selected_features = [
    'spkts', 'dpkts', 'sbytes', 'dbytes', 'sloss', 'dloss', 'swin', 'dwin', 'stcpb', 'dtcpb', 'dttl', 
    'proto', 'ct_state_ttl', 'service', 'state', 
    'ct_srv_dst', 'ct_dst_ltm', 'ct_srv_src', 'ct_src_ltm', 'ct_src_dport_ltm', 
    'ct_dst_sport_ltm', 'ct_dst_src_ltm'
]

# Ensure selected features exist in the dataset
selected_features = [f for f in selected_features if f in df.columns]

# Separate features and target
target_column = 'label'  # Replace with actual target column
X = df[selected_features]
y = df[target_column]

# One-hot encode categorical columns
X_encoded = pd.get_dummies(X)

# Handle missing values if any
X_encoded = X_encoded.fillna(0)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_encoded, y)

# Standardize features
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Define base estimator
base_est = DecisionTreeClassifier(max_depth=3)

# Set up AdaBoost classifier with specified hyperparameters
adaboost = AdaBoostClassifier(estimator=base_est, n_estimators=500, learning_rate=1, algorithm='SAMME', random_state=42)

# Train model
adaboost.fit(X_train, y_train)

# Predict and evaluate
y_pred = adaboost.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"AdaBoost Model Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Convert to percentages
cm_percent = cm.astype('float') / cm.sum() * 100

# Print percentage confusion matrix
print("\nConfusion Matrix  of AdaBoost (Percentage Form):")
print(np.round(cm_percent, 2))  # Optional: round to 2 decimal places

# Optional: Display confusion matrix plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent)
disp.plot(cmap='Blues', values_format='.2f')
plt.title("Confusion Matrix (Percentage)")
plt.show()
