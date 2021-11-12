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
from sklearn.ensemble import RandomForestClassifier


# import library
from sklearn.model_selection import GridSearchCV


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# drop "Loan_Status" and assign it to target variable
X = train.drop('Loan_Status', axis=1)
y = train.Loan_Status

#divide the dataset into train, test 
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)

# Provide range for max_depth from 1 to 20 with an interval of 2 and from 1 to 200 with an interval of 20 for n_estimators
paramgrid = {'max_depth': list(range(1, 20, 2)), 'n_estimators': list(range(1, 200, 20))}

# default 3-fold cross validation, cv=3
grid_search = GridSearchCV(RandomForestClassifier(random_state=1), paramgrid)
# split the data
from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X, y, test_size =0.3, random_state=1)

# fit the grid search model
grid_search.fit(x_train, y_train)
# estimate the optimized value
RF = grid_search.best_estimator_

def randomForest(X, xtest,y):
    mean_accuracy = []
    i=1
    kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
    for train_index,test_index in kf.split(X,y):
        print('\n{} of kfold {}'.format(i,kf.n_splits))
        xtr,xvl = X.loc[train_index],X.loc[test_index]
        ytr,yvl = y[train_index],y[test_index]
        
        model = RF
        model.fit(xtr, ytr)
        pred_test = model.predict(xvl)
        score = accuracy_score(yvl,pred_test)
        mean_accuracy.append(score)
        print('accuracy_score',score)
        i+=1
        
    print("\nMean validation accuracy: ", sum(mean_accuracy)/len(mean_accuracy))
    pred_test = model.predict(xtest)
    pred2=model.predict_proba(xtest)[:,1]
if __name__ == '__main__':
    randomForest(X, xtest,y)





