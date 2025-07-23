# TITANIC-PREDICTION
# Titanic Survival Prediction - CODSOFT Internship Task 1

## Objective
The goal of this project is to build a machine learning model that predicts whether a passenger survived the Titanic disaster based on various features such as age, gender, ticket class, and fare.

## Technologies Used
- Python
- Google Colab
- pandas, numpy
- seaborn, matplotlib
- scikit-learn (RandomForestClassifier, train_test_split, metrics)

## Dataset
- Source: [Kaggle Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- File used: `titanic.csv`

## Workflow
1. Uploaded dataset using Colab's file upload feature
2. Performed data cleaning (handled missing values, dropped irrelevant columns)
3. Applied label encoding on categorical features (Sex, Embarked)
4. Split data into training and testing sets
5. Trained a Random Forest Classifier
6. Evaluated the model using accuracy, confusion matrix, and classification report
7. Visualized feature importance to understand key predictors

## Results
- **Model Accuracy:** Approximately 81.33% (based on test set)

## File Structure
task1_titanic_survival_prediction/
├── titanic_prediction.ipynb # or .py if you used script
├── titanic.csv
└── README.md

## Author
KAMALEESHWARI P  
CODSOFT Data Science Intern - 2025

Data Science Internship Tasks for CodSoft
