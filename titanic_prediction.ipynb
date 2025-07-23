# CODSOFT Internship - Task 1: Titanic Survival Prediction
# Author: [KAMALEESHWARI P]

# Step 1: Upload the Titanic dataset
from google.colab import files
print("Please upload 'titanic.csv' file from your computer.")
uploaded = files.upload()

# Step 2: Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Step 3: Load dataset
data = pd.read_csv("titanic.csv")
print("First few rows of the dataset:")
print(data.head())

# Step 4: Data Cleaning and Preprocessing
# Drop irrelevant columns
data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

# Handle missing values
data["Age"].fillna(data["Age"].median(), inplace=True)
data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
data["Sex"] = label_encoder.fit_transform(data["Sex"])  # male = 1, female = 0
data["Embarked"] = label_encoder.fit_transform(data["Embarked"])

# Step 5: Feature and target separation
X = data.drop("Survived", axis=1)
y = data["Survived"]

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 7: Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Predictions and evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 9: Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Step 10: Feature importance visualization
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.sort_values().plot(kind="barh", figsize=(8, 5), color="darkcyan")
plt.title("Feature Importance")
plt.xlabel("Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
