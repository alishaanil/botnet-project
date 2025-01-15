from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('cleaned_data.csv')

# Separate the features and target variable
X = df.drop(columns=['label'])  # Replace 'target' with your actual target column
y = df['label']  # Target variable

# Initialize SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 , stratify=y)


# Apply SMOTE to the training data
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Check class distribution after SMOTE
#print("Class distribution in training set after SMOTE:")
#print(y_train_res.value_counts())

columns_to_scale = [col for col in X_train_res.columns if col != 'label']

# Applying Min-Max scaling only to the selected columns
scaler = MinMaxScaler()
X_resampled_scaled = X_train_res.copy()  # To avoid modifying the original DataFrame
X_resampled_scaled[columns_to_scale] = scaler.fit_transform(X_train_res[columns_to_scale])

# Optionally, scale the test set (but fit the scaler on the training set only)
X_test_scaled = X_test.copy()
X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])


# Initialize a Random Forest model
model = RandomForestClassifier(n_estimators=500 , min_samples_split=10 , min_samples_leaf=2 , max_features='sqrt' , max_depth=None , n_jobs=-1 )


# Train the model on the resampled (balanced) training data
model.fit(X_train_res, y_train_res)



# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
#print(classification_report(y_test, y_pred))

# Initialize Random Forest directly
#best_rf = RandomForestClassifier(random_state=42)

# Fit the Random Forest model to your training data
#best_rf.fit(X_train_res, y_train_res)

# Perform 5-fold cross-validation
#cv_scores = cross_val_score(best_rf, X_train_res, y_train_res, cv=5)

# Average cross-validation score
#print("Average 5-Fold CV Score: ", cv_scores.mean())

cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=5)

print(f'Cross-Validation Accuracy: {cv_scores.mean()}')

# Average cross-validation score
print("Average 5-Fold CV Score: ", cv_scores.mean())


# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Generating a classification report
print(classification_report(y_test, y_pred))

