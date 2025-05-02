# Diabetes---EDA-Prediction
Diabetes Prediction: A Comprehensive Analysis
Objective
Predict diabetes onset using diagnostic measurements via machine learning.
Dataset
Uses diabetes.csv (Pima Indians Diabetes Dataset) with diagnostic features and Outcome (0 = No Diabetes, 1 = Diabetes).
Methodology

Data Loading & Exploration: Loaded data with pandas, checked structure, missing values, and Outcome distribution.
EDA & Visualization: Plotted histograms, correlation heatmap, and Outcome count plot to analyze distributions and relationships.
Data Cleaning: Winsorized outliers, imputed zero values with medians, validated data types.
Data Splitting: Split into 80% training, 20% testing sets (train_test_split).
Feature Scaling: Standardized features using StandardScaler.
Model Training: Trained Logistic Regression model.
Model Evaluation: Calculated accuracy, precision, recall, F1-score, AUC-ROC; generated confusion matrix.
Model Optimization: Tuned hyperparameters with GridSearchCV (F1-score metric).

Key Findings

Data: No missing values; addressed outliers and zeros. Glucose correlated with Outcome.
Performance: Initial model accuracy ~0.76; optimized model: accuracy 0.7597, precision 0.6667, recall 0.6545, F1-score 0.6606, AUC-ROC 0.7364.
Best Hyperparameters: C=0.1, solver='liblinear'.

Next Steps

Test Random Forests, SVMs, or Gradient Boosting.
Explore feature engineering (e.g., interaction terms).
Address class imbalance with SMOTE or sampling.

Installation
pip install pandas numpy seaborn matplotlib scikit-learn

Usage

Clone repo: git clone https://github.com/your-username/diabetes-prediction.git
Add diabetes.csv to directory.
Run: jupyter notebook diabetes_prediction.ipynb

Dataset

Source: Kaggle
Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
Target: Outcome

License
MIT License. See LICENSE.
Acknowledgments
Inspired by Mousa et al. (2023) and open-source tools like scikit-learn and pandas.
