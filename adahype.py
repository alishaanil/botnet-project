import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score , explained_variance_score
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

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

# Convert categorical columns to numeric using one-hot encoding
X_encoded = pd.get_dummies(X)

# Handle missing values if any (optional)
X_encoded = X_encoded.fillna(0)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_encoded, y)

# Standardize features for AdaBoost
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Set up cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define AdaBoost model
adaboost_model = AdaBoostClassifier(n_estimators=50, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(adaboost_model, X_resampled, y_resampled, cv=cv, scoring='accuracy')
print(f'CV Scores: {cv_scores}')
print(f'Mean CV Score: {cv_scores.mean():.4f}')
print(f'Standard Deviation: {cv_scores.std():.4f}')

# Train the model
adaboost_model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = adaboost_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
variance_score = explained_variance_score(y_test, y_pred)
print(f"Variance Score: {variance_score:.4f}")



print(f"Best AdaBoost Model Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Generate classification report
report = classification_report(y_test, y_pred)
print("\nClassification Report:\n", report)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Visualize confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()