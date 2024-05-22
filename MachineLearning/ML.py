import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import data_cleaning



def run_ml():
    loan_df_train = pd.read_csv('../loan_data_set/loan-train.csv')
    loan_df_test = pd.read_csv('../loan_data_set/loan-test.csv')

    loan_df_train, loan_df_test = data_cleaning.clean_data(loan_df_train, loan_df_test)
    scaler = MinMaxScaler()
    features_to_scale = ['LoanAmount', 'ApplicantIncome', 'CoapplicantIncome']
    loan_df_train[features_to_scale] = scaler.fit_transform(loan_df_train[features_to_scale])
    loan_df_test[features_to_scale] = scaler.transform(loan_df_test[features_to_scale])

    # Prepare data for
    X = loan_df_train.drop('Loan_Status', axis=1)
    y = loan_df_train['Loan_Status'].apply(lambda x: 1 if x == 'Y' else 0)
    X = pd.get_dummies(X, drop_first=True)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    #initialize and train the decision tree forestclassifier

    dtc = DecisionTreeClassifier(max_depth=13)
    dtc.fit(X_train,y_train)


    # Predict on the test set
    y_pred = rf.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy for RandomForest is: {accuracy:.2f}')
    print(classification_report(y_test, y_pred))
    y_pred_dtc = dtc.predict(X_test)

# Evaluate the classifier
    accuracy_dtc = accuracy_score(y_test, y_pred_dtc)
    print(f'Accuracy for DecisionTreeClassifier is: {accuracy_dtc:.2f}')
    print(classification_report(y_test, y_pred_dtc))
    # Evaluate using core variables only
    X_core = loan_df_train[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']]
    y_core = loan_df_train['Loan_Status']#.apply(lambda x: 1 if x == 'Y' else 0)

    X_train_core, X_test_core, y_train_core, y_test_core = train_test_split(X_core, y_core, test_size=0.1,
                                                                            random_state=42)
    rf_core = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_core.fit(X_train_core, y_train_core)
    y_pred_core = rf_core.predict(X_test_core)
    accuracy_core = accuracy_score(y_test_core, y_pred_core)
    print(f'Accuracy for three core variables is: {accuracy_core:.2f}')
    print(classification_report(y_test_core, y_pred_core))
# Join the result to loan-test
    X_test_final = pd.get_dummies(loan_df_test, drop_first=True)
    y_test_pred = rf.predict(X_test_final)

    # Append predictions to the test dataframe
    loan_df_test['Predicted_Loan_Status'] = y_test_pred
    loan_df_test['Predicted_Loan_Status'] = loan_df_test['Predicted_Loan_Status'].apply(lambda x: 'Y' if x == 1 else 'N')

    # Optionally save or display the updated dataframe
    loan_df_test.to_csv('../loan_data_set/loan-test-predictions.csv', index=False)
    print(loan_df_test.head())

