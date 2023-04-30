import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
# -------------------------- Models -------------------------- #
from sklearn.neighbors import KNeighborsClassifier
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


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.16, train_size=0.84, random_state=42)

# -------------------------- Models -------------------------- #
highest = 0
best = 0
best_clf = None
for n in range(1, 50):
    clf = KNeighborsClassifier(n_neighbors=n, weights='distance')
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    if score > highest:
        highest = score
        best = n
        best_clf = clf
        print("New highest: {highest} with {best} neighbors".format(highest=highest, best=best))
with open('models/knn/{}nn.pkl'.format(best), 'wb') as f:
    pickle.dump(best_clf, f)
    
print(highest, best)

    
'''
# -------------------------- Plotting -------------------------- #
'''
