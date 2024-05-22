import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import data_cleaning
import os


def run_eda():
    # EDA
    train = pd.read_csv('../loan_data_set/loan-train.csv')
    test = pd.read_csv('../loan_data_set/loan-test.csv')

    train_cleaned, test_cleaned = data_cleaning.clean_data(train, test)
    data = data_cleaning.encode_data(train_cleaned)
    print("General Descriptive Statistics:")
    print(data.describe().to_string())

    # Descriptive statistics by Education level
    print("\nDescriptive Statistics by Education Level:")
    print(data.groupby('Education').describe().to_string())

    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    # Distribution of Loan Amounts by Education level
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Education', y='LoanAmount', data=data)
    plt.title('Distribution of Loan Amounts by Education Level')
    plt.xticks(ticks=[0, 1], labels=['Not Graduate', 'Graduate'])
    plt.xlabel('Education Level')
    plt.ylabel('Loan Amount')
    plt.savefig(os.path.join('../loan_data_set', 'Education-loan_amount.png'))
    #plt.show()

    # Relationship between Applicant Income and Loan Amount by Education level
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='ApplicantIncome', y='LoanAmount', hue='Education', data=data, style='Education')
    plt.title('Applicant Income vs. Loan Amount by Education Level')
    plt.xticks(ticks=[0, 1], labels=['Not Graduate', 'Graduate'])
    plt.xlabel('Applicant Income')
    plt.ylabel('Loan Amount')
    plt.legend(title='Education Level', labels=['Not Graduate', 'Graduate'])
    plt.savefig(os.path.join('../loan_data_set', 'ApplicantIncome-loan_amount.png'))
    #plt.show()

    # Checking for any relationships with Coapplicant Income as well
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='CoapplicantIncome', y='LoanAmount', hue='Education', data=data, style='Education')
    plt.title('Coapplicant Income vs. Loan Amount by Education Level')
    plt.xlabel('Coapplicant Income')
    plt.ylabel('Loan Amount')
    plt.legend(title='Education Level', labels=['Not Graduate', 'Graduate'])
    plt.savefig(os.path.join('../loan_data_set', 'CoapplicantIncome-loan_amount.png'))
    #plt.show()


