import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

train=pd.read_csv('https://raw.githubusercontent.com/limchiahooi/loan-approval-prediction/master/train_u6lujuX_CVtuZ9i.csv')
test = pd.read_csv('https://datahack-prod.s3.amazonaws.com/test_file/test_lAUu6dG.csv')




def preprocessing(train, test):

    train['Dependents'].replace('3+', 3, inplace=True)
    test['Dependents'].replace('3+', 3, inplace=True)

    # replacing Y and N in Loan_Status variable with 1 and 0 respectively
    train['Loan_Status'].replace('N', 0, inplace=True)
    train['Loan_Status'].replace('Y', 1, inplace=True)
    
    # replace missing values with the mode

    
    train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
    train['Married'].fillna(train['Married'].mode()[0], inplace=True)
    train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
    train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
    train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
    train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
    train["LoanAmount"].fillna(train['LoanAmount'].median(), inplace=True)

    #outlier treatment
    train['loanAmount_log'] = np.log(train['LoanAmount'])
    test['loanAmount_log'] = np.log(test['LoanAmount'])

    train = pd.get_dummies(train)
    test = pd.get_dummies(test)

    #Feature Engineering 
    # combine Applicant Income and Coapplicant Income into a new variable
    train['Total_Income'] = train['ApplicantIncome'] + train['CoapplicantIncome']
    test['Total_Income'] = test['ApplicantIncome'] + test['CoapplicantIncome']
    # log transformation
    train['Total_Income_log'] = np.log(train['Total_Income'])
    test['Total_Income_log'] = np.log(test['Total_Income'])

    # create EMI feature
    train['EMI'] = train['LoanAmount'] / train['Loan_Amount_Term']
    test['EMI'] = test['LoanAmount'] / test['Loan_Amount_Term']


    # create new "Balance Income" variable
    train['Balance Income'] = train['Total_Income'] - (train['EMI']*1000) # Multiply with 1000 to make the units equal 
    test['Balance Income'] = test['Total_Income'] - (test['EMI']*1000)

    # drop the variables
    train = train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)
    test = test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)
    train.to_csv('./train.csv')
    test.to_csv('./test.csv')


if __name__ == '__main__':
    preprocessing(train, test)




