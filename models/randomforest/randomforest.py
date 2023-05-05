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
from sklearn.model_selection import train_test_split, cross_val_score



# -------------------------- Data -------------------------- #
dataframe = pd.read_csv('data/patil/dataset.csv').dropna(axis=1).sample(frac=1)
x = dataframe.drop("Disease", axis=1)
y = dataframe["Disease"]
#print(dataframe.drop('Disease', axis=1))
le = LabelEncoder()
oe = OrdinalEncoder()
y = le.fit_transform(y)
x = oe.fit_transform(x)



#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.16, train_size=0.84)

# -------------------------- Models -------------------------- #
best_accuracy = 0
best_estimators = 0
best_depth = 0
best_criteria = ""
best_bootstap = False
best_min_samples_split = 0
best_model = None

with open('models/randomforest/randomforest.json', 'r') as f:
    data = json.load(f)
    best_accuravy = data['accuracy']
    best_estimators = data['estimators']
    best_depth = data['depth']
    best_criteria = data['criteria']
    best_bootstap = data['bootstrap']
with open('models/randomforest/randomforest.pkl', 'rb') as f:
    best_model = pickle.load(f)   

for n in range (1, 50):
    for max in range(1, 20):
        for criterion in ['gini', 'entropy', 'log_loss']:
            for bootstap in [True, False]:
                print("Attempting with {n} estimators, {max} depth, {criterion} criterion, and {bootstap} bootstap".format(n=n, max=max, criterion=criterion, bootstap=bootstap))
                clf = RandomForestClassifier(n_estimators=n, max_depth=max, criterion='entropy', bootstrap=bootstap)
                score = np.mean(cross_val_score(clf, x, y, cv=6, scoring='recall_macro'))
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

