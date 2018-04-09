import sklearn
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import load_titanic

def expand_df(X_train,T_train,n=500):    
    x1 = np.where(T_train==1)
    x0 = np.where(T_train==0)
    
    x1 = X_train.iloc[x1].sample(n,replace=True)
    x0 = X_train.iloc[x0].sample(n,replace=True)
    X = pd.DataFrame(columns=X_train.columns)
    X = X.append(x0).append(x1)
    X['PassengerId'] = X['PassengerId']+1000
    X = X.append(X_train).as_matrix()
    T = np.concatenate((np.zeros(n),np.ones(n)))
    T = np.concatenate((T,T_train.as_matrix()))
    return X,T

X_train, T_train, X_test, T_test = load_titanic.load(True)

n = 5
Times = np.zeros(n)
for i in range(1,n+1):
    start = time.time()
    clf = RandomForestClassifier(n_estimators=10, max_features=2)
    clf.fit(X_train,T_train)
    end = time.time()
    Times[i-1] = end-start
print("RandomForest", Times)

n=6
Times = np.zeros(n+1)
for i in range(1,n+1):
    clf = DecisionTreeClassifier(criterion='entropy')
    X,T = expand_df(X_train,T_train,10**i)
    start = time.time()
    print(X.shape)
    clf.fit(X,T)
    end = time.time()
    Times[i] = end-start

print("Decision Tree", Times)
