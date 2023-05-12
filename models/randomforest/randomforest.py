import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
# -------------------------- Models -------------------------- #
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
# -------------------------- Preprocessing -------------------------- #
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, TargetEncoder, OneHotEncoder
# -------------------------- Metrics -------------------------- #
from sklearn.model_selection import train_test_split, cross_val_score


# -------------------------- Data -------------------------- #
df = pd.read_csv('data/patil/dataset.csv').sample(frac=1)
x = df.drop("Disease", axis=1)
y = df["Disease"]
#print(dataframe.drop('Disease', axis=1))
le = LabelEncoder()
#te = TargetEncoder(target_type='binary', cv=4)
one_e = OneHotEncoder()

y = le.fit_transform(y)
one_e.fit(x)
x = one_e.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.4, train_size=.6)
#x_train = te.fit_transform(x_train, y_train)
#x_test = te.transform(x_test)
# -------------------------- Models -------------------------- #
best_accuracy = 0
best_estimators = 0
best_depth = 0
best_criteria = ""
best_bootstap = False
best_min_samples_split = 0
best_model = None
'''
try:
    with open('models/randomforest/randomforest.json', 'r') as f:
        data = json.load(f)
        best_accuravy = data['accuracy']
        best_estimators = data['estimators']
        best_depth = data['depth']
        best_criteria = data['criteria']
        best_bootstap = data['bootstrap']
    with open('models/randomforest/randomforest.pkl', 'rb') as f:
        best_model = pickle.load(f)
except Exception as e:
    print(e)
'''
for n in range (1, 20):
    for max in range(1, 17):
        for criterion in ['entropy', 'log_loss']:
            for bootstap in [True, False]:
                print("Attempting with {n} estimators, {max} depth, {criterion} criterion, and {bootstap} bootstap".format(n=n, max=max, criterion=criterion, bootstap=bootstap))
                clf = RandomForestClassifier(n_estimators=n, max_depth=max, criterion='entropy', bootstrap=bootstap)
                score = cross_val_score(clf, x, y, cv=6).mean()
                #clf.fit(x_train, y_train)
                #score = clf.score(x_test, y_test)
                if score > best_accuracy: 
                    best_accuracy = score
                    best_estimators = n
                    best_depth = max
                    best_criteria = criterion
                    best_bootstap = bootstap
                    best_model = clf
                    print("New highest: {highest} with {n} estimators and {m} depth".format(highest=best_accuracy, n=n, m=max))
with open('models/randomforest/randomforest.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('models/randomforest/randomforest.json', 'w') as f:
    json.dump({
        "accuracy": best_accuracy,
        "estimators": best_estimators,
        "depth": best_depth,
        "criteria": best_criteria,
        "bootstrap": best_bootstap
    }, f)
    

# -------------------------- Plotting -------------------------- #

