"""
This file is solely responsible for exporting the results after data cleaning, omitting any irrelevant steps.
Specific analyses are conducted in data_preparation.ipynb and report.pdf
"""

import pandas as pd

def clean_data(train, test):
    # Drop unecessary variables
    train = train.drop(['Loan_ID'], axis=1)
    test = test.drop(['Loan_ID'], axis=1)

    # Impute the missing values
    train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
    train['Married'].fillna(train['Married'].mode()[0], inplace=True)
    train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
    train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
    train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
    train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
    train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

    test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
    test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
    test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
    test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
    test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
    test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

    return train, test

def encode_data(data):
    data['Dependents'] = data['Dependents'].replace('3+', '3')
    if data['Dependents'].dtype == 'object':
        data['Dependents'] = pd.to_numeric(data['Dependents'])

    # convert 'Education' to a binary variable where 'Graduate'=1 and 'Not Graduate'=0
    data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})

    return data
