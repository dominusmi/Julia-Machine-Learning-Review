import pandas as pd
import numpy as np

def convert_embarked(loc):
    if loc == "S":
        return 0
    elif loc == "C":
        return 1
    else:
        return 2

def load(as_matrix=False):
    """Loads titanic data and does some pre-processing
        use load() for pandas data frames and load(True) 
        for matrix conversion."""
    titanic = pd.read_csv("resources/titanic_train.csv")
    # drop some cols first
    titanic = titanic.drop(['Ticket','Cabin','Name'],axis=1)
    # removes the NaN values
    titanic = titanic.drop(titanic.index[[np.where(titanic.isnull())[0]]],axis=0)
    # sample
    test = titanic.sample(80)
    train = titanic.drop(test.index,axis=0)

    # split into features and targets
    test_targets = test['Survived']
    train_targets = train['Survived']
    test = test.drop(['Survived'],axis=1)
    train = train.drop(['Survived'],axis=1)
    
    # the encoding, much nicer using Julia's .
    for i in train.index:
        train.at[i,'Embarked']=convert_embarked(train['Embarked'][i])
        if (train['Sex'][i] == "male"):
            train.at[i, 'Sex'] = 0
        else:
            train.at[i, 'Sex']=1
    for i in test.index:
        test.at[i,'Embarked']= convert_embarked(test['Embarked'][i])
        if (test['Sex'][i] == "male"):
            test.at[i, 'Sex']=0
        else:
            test.at[i, 'Sex']=1
    # return raw matrices 
    if(as_matrix):
        return train.as_matrix(), train_targets.as_matrix(), test.as_matrix(), test_targets.as_matrix()
    # return the pandas data frames
    else:
        return train, train_targets, test, test_targets
