import sklearn
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import load_titanic

X_train, T_train, X_test, T_test = load_titanic.load(True)

n = 5
Times = np.zeros(n)
for i in range(1,n+1):
    start = time.time()
    clf = RandomForestClassifier(n_estimators=10, max_features=2)
    clf.fit(X_train,T_train)
    end = time.time()
    Times[i-1] = end-start
print(Times)
