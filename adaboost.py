import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint




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

# Train an AdaBoost classifier
adaboost_model = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost_model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = adaboost_model.predict(X_test)

# Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"AdaBoost Model Accuracy: {accuracy:.4f}")

# Generate and print classification report
report = classification_report(y_test, y_pred)
print("\nClassification Report:\n")
print(report)


