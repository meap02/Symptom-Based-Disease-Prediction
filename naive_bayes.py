import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
import csv

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import f1_score, accuracy_score

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from sklearn.model_selection import train_test_split, cross_val_score

dataframe = pd.read_csv('data/patil/dataset.csv').dropna(axis=1).sample(frac=1)
x = dataframe.drop("Disease", axis=1)
y = dataframe["Disease"]

le = LabelEncoder()
oe = OrdinalEncoder()
y = le.fit_transform(y)
x = oe.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.16, train_size=0.84, random_state=42)

print(x_test[5])

model = GaussianNB()
model.fit(x_train, y_train)

predicted = model.predict(x_test)

print("F1-score %", f1_score(y_test, predicted, average='macro')*100, "\n" 'Accuracy% =', accuracy_score(y_test, predicted)*100 )