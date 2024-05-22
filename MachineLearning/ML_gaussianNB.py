import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import data_cleaning


def run_gaussian_nb():
    loan_df_train = pd.read_csv('../loan_data_set/loan-train.csv')
    loan_df_test = pd.read_csv('../loan_data_set/loan-test.csv')

    loan_df_train, loan_df_test = data_cleaning.clean_data(loan_df_train, loan_df_test)
    scaler = StandardScaler()
    features_to_scale = ['LoanAmount', 'ApplicantIncome', 'CoapplicantIncome']
    loan_df_train[features_to_scale] = scaler.fit_transform(loan_df_train[features_to_scale])
    loan_df_test[features_to_scale] = scaler.transform(loan_df_test[features_to_scale])

    X = loan_df_train.drop('Loan_Status', axis=1)
    y = loan_df_train['Loan_Status'].apply(lambda x: 1 if x == 'Y' else 0)
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred_gnb = gnb.predict(X_test)

    accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
    print(f'Accuracy for GaussianNB is: {accuracy_gnb:.2f}')
    print(classification_report(y_test, y_pred_gnb))

