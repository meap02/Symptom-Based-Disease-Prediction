import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
# -------------------------- Models -------------------------- #
from sklearn.ensemble import RandomForestClassifier
# -------------------------- Preprocessing -------------------------- #
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
# -------------------------- Metrics -------------------------- #
from sklearn.model_selection import train_test_split



# -------------------------- Data -------------------------- #
dataframe = pd.read_csv('data/patil/dataset.csv').dropna(axis=1).sample(frac=1)
x = dataframe.drop("Disease", axis=1)
y = dataframe["Disease"]
#print(dataframe.drop('Disease', axis=1))
le = LabelEncoder()
oe = OrdinalEncoder()
y = le.fit_transform(y)
x = oe.fit_transform(x)




# -------------------------- Models -------------------------- #
highest_accuracy = 0
best_estimators = 0
best_depth = 0
best_clf = None
for n in range (1, 100):
    for m in range(1, 20):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.16, train_size=0.84)
        clf = RandomForestClassifier(n_estimators=n, max_depth=m, criterion='entropy')
        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        if score > highest_accuracy: 
            highest_accuracy = score
            best_estimators = n
            best_depth = m
            best_clf = clf
            print("New highest: {highest} with {n} estimators and {m} depth".format(highest=highest_accuracy, n=n, m=m))
with open('models/randomforest/randomforest.pkl', 'wb') as f:
    pickle.dump(best_clf, f)
with open('models/randomforest/randomforest.json', 'w') as f:
    json.dump({"accuracy":highest_accuracy, "estimators": best_estimators, "depth": best_depth}, f)
    
'''
# -------------------------- Plotting -------------------------- #
'''
