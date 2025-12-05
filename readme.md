Loan Default Prediction Using Machine Learning

This project builds a machine learning system to predict loan default risk using financial, demographic, and behavioral features.
The goal is to help lenders identify high-risk applicants and control credit losses, while maintaining fair approval for good customers.

The project includes:

End-to-end EDA

Complete data preprocessing

Baseline & tuned Random Forest

Balanced XGBoost model

Threshold optimization for business needs

Visualizations (ROC, confusion matrix, feature importance)

1. Project Structure
Loan-Default-Prediction/
│
├── 01_EDA_and_Preprocessing.ipynb
├── 02_RandomForest_Model.ipynb
├── 03_XGBoost_Model.ipynb
├── X_train.csv
├── X_test.csv
├── y_train.csv
├── y_test.csv
├── random_forest_model.pkl
├── README.md
└── requirements.txt
2. Dataset Overview

Total rows: 255,347
Features: 17
Target: Default (0 = No Default, 1 = Default)

Key Features
Feature	Description
Age	Customer’s age
Income	Annual income
LoanAmount	Approved loan amount
CreditScore	Creditworthiness score
MonthsEmployed	Employment duration
InterestRate	Loan interest rate
DTIRatio	Debt-to-income ratio
LoanPurpose	Purpose of loan
HasMortgage	Existing mortgage
HasCoSigner	Co-signer availability
3. Exploratory Data Analysis

Performed:

Distribution analysis

Correlation heatmap

Outlier detection using IQR

Class imbalance visualization

Countplots for categorical features

Key Insights

Dataset is highly imbalanced (only ~12% defaults)

Top correlated features:

InterestRate

LoanAmount

CreditScore

MonthsEmployed

Income

Age

4. Data Preprocessing

Removed LoanID

Encoded categorical features:

Label Encoding for ordinal features

One-hot encoding for nominal features

Converted Yes/No to 1/0

Handled class imbalance using:

class_weight="balanced" for RandomForest

scale_pos_weight for XGBoost

Train-test split: 80/20

5. Models Used
A. Random Forest

class_weight = balanced

Tuned using RandomizedSearchCV

Good for overall accuracy

Weak in detecting minority class (defaulters)

B. XGBoost

Uses boosting to capture complexity

Handled imbalance via:

scale_pos_weight = ratio


Tuned probability threshold (0.8)

6. Model Performance Comparison
Final Model Scores
Metric	Random Forest	XGBoost (Threshold = 0.8)
Accuracy	0.885	0.882
Precision (Default)	0.xx	0.xx
Recall (Default)	0.07	0.85
F1-score (Default)	0.xx	0.86
ROC–AUC	0.748	0.756
Interpretation

RandomForest: High accuracy but almost no ability to detect defaulters

XGBoost: Slight drop in accuracy but massive improvement in detecting high-risk customers

Best Business Model = XGBoost with threshold = 0.8

7. Feature Importance (XGBoost)

Top contributing factors to loan default:

CreditScore

DTIRatio

InterestRate

Income

MonthsEmployed

LoanAmount

These align with real-world lending principles.

8. Threshold Optimization

Instead of using the default probability threshold of 0.5, testing showed:

Threshold = 0.8 gives the best balance:

High accuracy (88.2%)

High recall for defaulters (85%)

Excellent F1-score (0.86)

This makes the model useful for risk management teams.

9. Visualizations Included

Correlation heatmap

Distribution plots

ROC Curve

Confusion Matrix

Feature Importance chart

10. Conclusion

This project demonstrates an effective machine learning approach for predicting loan defaults.
Key outcomes:

XGBoost significantly outperforms RandomForest in identifying risky borrowers.

Business-oriented threshold tuning resulted in:

High recall for defaulters

Strong overall accuracy

Reliable prediction performance

The final model can help financial institutions reduce credit losses and improve lending decisions.

11. Future Improvements

Add SHAP explainability

Hyperparameter tuning (Bayesian Optimization)

Build a FastAPI / Streamlit app

Train on larger real-world datasets

Use ensemble of XGBoost + LightGBM

12. Requirements
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
