from scipy.stats import normaltest, mannwhitneyu, levene, linregress
import pandas as pd
import numpy as np
import data_cleaning
import seaborn as sns
import matplotlib.pyplot as plt


def run_stat():
    train = pd.read_csv('../loan_data_set/loan-train.csv')
    test = pd.read_csv('../loan_data_set/loan-test.csv')

    train_cleaned, test_cleaned = data_cleaning.clean_data(train, test)
    data = data_cleaning.encode_data(train_cleaned)

    # Grouping Education and LoanAmount
    graduate = data[data['Education'] == 1]['LoanAmount']
    N_graduate = data[data['Education'] == 0]['LoanAmount']

    sns.set(style='whitegrid')
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    sns.histplot(graduate, kde=True)
    plt.title('Graduated Loan Amount Distribution')
    plt.xlabel('Loan Amount')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    sns.histplot(N_graduate, kde=True)
    plt.title('Not Graduated Loan Amount Distribution')
    plt.xlabel('Loan Amount')
    plt.ylabel('Frequency')

    plt.tight_layout()
    #plt.show()

    # Normality Testing
    # Test if their relation is normally distributed
    gra_stat_n, gra_p_val_n = normaltest(graduate)
    print(f'Testing normality of Graduated using normaltest: stat = {gra_stat_n}, p-value = {gra_p_val_n}')
    N_gra_stat_n, N_gra_p_val_n = normaltest(N_graduate)
    print(f'Testing normality of Not Graduated using normaltest: stat = {N_gra_stat_n}, p-value = {N_gra_p_val_n}')

    print('Normal testing method shows that the dataset is not normally distributed.')

    # Equal Variance Testing
    stat_lev, p_val_lev = levene(graduate, N_graduate)
    print(f'Testing Equal Variance using levene: stat = {stat_lev}, p-value = {p_val_lev}')

    # Testing if taking log to the dataframe improves the p-value
    graduate_log = np.log(graduate)
    N_graduate_log = np.log(N_graduate)
    gra_log, gra_log_p = normaltest(graduate_log)
    print(f'Testing normality of log of Graduated using normaltest: stat = {gra_log}, p-value = {gra_log_p}')
    N_gra_log, N_log_p = normaltest(N_graduate_log)
    print(f'Testing normality of log of Not Graduated using normaltest: stat = {N_gra_log}, p-value = {N_log_p}')

    # Mann-Whitney U test
    # H0: Education and Loan Amount has no relation
    u_stat, u_p_val = mannwhitneyu(graduate_log, N_graduate_log)
    print(f'U test statistics={u_stat}, p-value={u_p_val}')

    # Testing the relation of Applicant Income and Loan Amount
    highavg_inc = data[data['ApplicantIncome'] > 5400]['LoanAmount']
    lowavg_inc = data[data['ApplicantIncome'] < 5400]['LoanAmount']

    high_stat, high_p_val = normaltest(highavg_inc)
    print(f'Testing normality of Graduated using normaltest: stat = {high_stat}, p-value = {high_p_val}')
    low_stat, low_p_val = normaltest(lowavg_inc)
    print(f'Testing normality of Not Graduated using normaltest: stat = {low_stat}, p-value = {low_p_val}')

    print('Normal testing method shows that the dataset is not normally distributed.')

    # Testing if taking log to the dataframe improves the p-value
    high_log = np.log(highavg_inc)
    low_log = np.log(lowavg_inc)
    h_log, h_log_p = normaltest(high_log)
    print(f'Testing normality of log of high income using normaltest: stat = {h_log}, p-value = {h_log_p}')
    l_log, l_log_p = normaltest(low_log)
    print(f'Testing normality of log of low income using normaltest: stat = {l_log}, p-value = {l_log_p}')

    # H0: Applicant Income and Loan Amount has no relation
    # seeking to reject this H0
    u_stat, u_p_val = mannwhitneyu(highavg_inc, lowavg_inc)
    print(f'U test statistics={u_stat}, p-value={u_p_val}')


