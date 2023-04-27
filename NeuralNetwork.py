import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def accuracy(confusionMatrix):
    diagonal_sum = confusionMatrix.trace()
    sum_of_all_elements = confusionMatrix.sum()
    return diagonal_sum / sum_of_all_elements


dataframe = pd.read_csv('data/patil/dataset.csv').dropna(axis=1).sample(frac=1)
x = dataframe.drop("Disease", axis=1)
y = dataframe["Disease"]
le = LabelEncoder()
oe = OrdinalEncoder()
y = le.fit_transform(y)
x = oe.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.16, train_size=0.84)

clf = MLPClassifier(max_iter=450)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
cm = confusion_matrix(y_pred, y_test)
roundedAcc = round(100 * accuracy(cm), 2)
print("Accuracy of Neural Network: " + str(roundedAcc))
