import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def main():
    # Load the data
    loan_df_train = pd.read_csv('../loan_data_set/loan-train.csv')
    loan_df_test = pd.read_csv('../loan_data_set/loan-test.csv')


    # Define the independent variables to test
    variables = ['Age', 'Income', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']

    # Normalize the variables
    scaler = StandardScaler()
    loan_df_train[variables] = scaler.fit_transform(loan_df_train[variables])

    # Split the dataset
    X = loan_df_train[variables]  # Independent variables
    y = loan_df_train['LoanAmount']  # Dependent variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Dictionary to store model performance
    model_performance = {}

    # Iterate over each variable to fit a model
    for var in variables:
        # Preparing the data - reshaping for a single feature
        X_train_var = X_train[[var]]
        X_test_var = X_test[[var]]

        # Create and fit the model
        model = LinearRegression(fit_intercept=False)
        model.fit(X_train_var, y_train)

        # Evaluate the model
        r2 = model.score(X_test_var, y_test)

        # Store the coefficient and R² score
        model_performance[var] = {'Coefficient': model.coef_[0], 'Score': r2}

    for var, performance in model_performance.items():
        print(f"Variable: {var}, Coefficient: {performance['Coefficient']}, Score: {performance['Score']}")

if __name__ == "__main__":
    main()
