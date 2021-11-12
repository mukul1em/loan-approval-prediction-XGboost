import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

model_lr = LogisticRegression()


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# drop "Loan_Status" and assign it to target variable
X = train.drop('Loan_Status', axis=1)
y = train.Loan_Status



def logisticsReg(X,y):
    mean_accuracy = []
    i = 1
    kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

    for train_index, test_index in kf.split(X, y):
        print('\n{} of kfold {}'.format(i, kf.n_splits))
        xtr, xvl = X.loc[train_index], X.loc[test_index]
        ytr, yvl = y[train_index], y[test_index]
        
        model = LogisticRegression(random_state=1)
        model.fit(xtr, ytr)
        pred_test = model.predict(xvl)
        score = accuracy_score(yvl, pred_test)
        mean_accuracy.append(score)
        print('accuracy_score', score)
        i+=1
        
    print("\nMean validation accuracy: ", sum(mean_accuracy)/len(mean_accuracy))


if __name__ == '__main__':
    logisticsReg(X,y)






