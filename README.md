# Customer Churn Prediction

This project uses machine learning to predict customer churn based on service usage, contract type, billing method, and demographic details from the Telco Customer dataset.

## Dataset

- **Source**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Records**: 7032 customers
- **Target**: `Churn` (Yes/No)

## Project Goals

- Predict whether a customer is likely to churn
- Identify the most important features influencing churn
- Compare different classification models

## Key Steps

- Data cleaning (handling TotalCharges nulls)
- Label and one-hot encoding for categorical features
- Train-test split with stratification
- Model training: Logistic Regression & Random Forest
- Evaluation using accuracy, recall, and F1-score
- Feature importance visualization

## Results

| Model               | Accuracy | Recall (Churn) | F1-Score (Churn) |
|---------------------|----------|----------------|------------------|
| Logistic Regression | 79.9%    | 57%            | 60%              |
| Random Forest       | 78.5%    | 50%            | 55%              |

> Logistic Regression performed best on recall and F1 for churn class.

## Feature Insights

Top predictors of churn:
- TotalCharges
- Tenure
- MonthlyCharges
- Contract Type
- Internet Service Type
- Payment Method

## Folder Structure

customer-churn-prediction/ ├── models/ │ ├── logistic_model.pkl │ └── random_forest_model.pkl ├── churn_prediction.ipynb ├── WA_Fn-UseC_-Telco-Customer-Churn.csv ├── README.md


## Tools Used

- Python
- Pandas, NumPy
- scikit-learn, joblib
- Matplotlib, Seaborn

## Author

Prakash Sharma
