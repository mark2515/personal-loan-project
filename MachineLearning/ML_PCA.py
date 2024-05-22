import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import data_cleaning
def run_pca():
    loan_df_train = pd.read_csv('../loan_data_set/loan-train.csv')
    loan_df_test = pd.read_csv('../loan_data_set/loan-test.csv')

    loan_df_train, loan_df_test = data_cleaning.clean_data(loan_df_train, loan_df_test)


# Prepare X from loan_df with only numerical data (no need for get_dummies here)
    X = loan_df_train[['ApplicantIncome','CoapplicantIncome','LoanAmount']]


# Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

# Apply PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)


# Corrected: Print explained variance ratios
    print(f"Explained variance by component for loan_df: {pca.explained_variance_ratio_}")

