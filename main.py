import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# -------------------------- Models -------------------------- #
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
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

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.16, train_size=0.84, random_state=42)
# -------------------------- Models -------------------------- #
tree = DecisionTreeClassifier(criterion='entropy', random_state=42)
scores = cross_val_score(tree, x, y, cv=6, scoring='recall_macro')
print(scores)
