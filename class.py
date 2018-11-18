import pandas as pd
import numpy as np
def CV(df, classifier, nfold, norm=True):

    acc = []
    for i in range(nfold):
        y = df['class']
        train, test = stratified_split(y)

        if norm:
            X_train = norm_df(df.iloc[train, 0:8])
            X_test = norm_df(df.iloc[test, 0:8])
        else:
            X_train = df.iloc[train, 0:8]
            X_test = df.iloc[test, 0:8]

        y_train = y[train]
        y_test = y[test]

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        acc.append(accuracy(y_test, y_pred))

    return acc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
logreg = LogisticRegression()
rf = RandomForestClassifier()
dat = pd.read_csv('indians-diabetes.csv')
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = pd.read_csv(url, names=names)
res = CV(df,rf, 10, norm=False)
