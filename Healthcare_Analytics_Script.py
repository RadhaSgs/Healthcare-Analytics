
# Healthcare Analytics: Diabetes Prediction

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

# Load the dataset
dataset_path = "healthcare_diabetes_dataset.csv"  # Update this with the correct dataset path
data = pd.read_csv(dataset_path)

# ---------------------
# STEP 1: Data Exploration and Cleaning
# ---------------------
print("Data Head:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Drop rows/columns with excessive missing values
data = data.dropna()

# Data types
print("\nData Types:")
print(data.dtypes)

# Encode categorical variables if needed
if data.select_dtypes(include='object').shape[1] > 0:
    data = pd.get_dummies(data, drop_first=True)

# ---------------------
# STEP 2: Exploratory Data Analysis (EDA)
# ---------------------
# Pairplot for basic variable relationships
sns.pairplot(data, diag_kind='kde')
plt.title("Pairplot of Variables")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ---------------------
# STEP 3: Predictive Modeling
# ---------------------
# Define features (X) and target (y)
target_variable = "Outcome"  # Replace with the actual target column in your dataset
X = data.drop(columns=[target_variable])
y = data[target_variable]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)
y_pred_prob = rf_model.predict_proba(X_test)[:, 1]

# ---------------------
# STEP 4: Model Evaluation
# ---------------------
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC-AUC Score: {roc_auc:.2f}")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# ---------------------
# STEP 5: Insights and Recommendations
# ---------------------
# Feature Importance
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(feature_importances)

# Save feature importances as a CSV
feature_importances.to_csv("feature_importances.csv", index=False)

print("The project analysis and modeling are complete!")
