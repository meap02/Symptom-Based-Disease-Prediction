import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
import csv
from statistics import mean 

# -------------------------- Models -------------------------- #
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB

# -------------------------- Preprocessing -------------------------- #
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# -------------------------- Metrics -------------------------- #
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, accuracy_score

# -------------------------- Data -------------------------- #
dataframe = pd.read_csv('data/patil/dataset.csv').dropna(axis=1).sample(frac=1)
x = dataframe.drop("Disease", axis=1)
y = dataframe["Disease"]

le = LabelEncoder()
oe = OrdinalEncoder()
y = le.fit_transform(y)
x = oe.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.16, train_size=0.84, random_state=42)


# -------------------------- Models -------------------------- #
GNBclf = GaussianNB()
MNBclf = MultinomialNB()
CNBclf = ComplementNB()
BNBclf = BernoulliNB()
CatNBclf = CategoricalNB()

GNBclf.fit(x_train, y_train)
MNBclf.fit(x_train, y_train)
CNBclf.fit(x_train, y_train)
BNBclf.fit(x_train, y_train)
CatNBclf.fit(x_train, y_train)

predictedGNB = GNBclf.predict(x_test)
predictedMNB = MNBclf.predict(x_test)
predictedCNB = CNBclf.predict(x_test)
predictedBNB = BNBclf.predict(x_test)
predictedCatNB = CatNBclf.predict(x_test)

recallGNBclf = cross_val_score(GNBclf, x, y, cv=6, scoring='recall_macro')
recallMNBclf = cross_val_score(MNBclf, x, y, cv=6, scoring='recall_macro')
recallCNBclf = cross_val_score(CNBclf, x, y, cv=6, scoring='recall_macro')
recallBNBclf = cross_val_score(BNBclf, x, y, cv=6, scoring='recall_macro')
recallCatNBclf = cross_val_score(CatNBclf, x, y, cv=6, scoring='recall_macro')


print("\n--Gaussian Naive Bayes--")
print("F1-score = ", f1_score(y_test, predictedGNB, average='macro')*100, "%\n" 'Accuracy =', accuracy_score(y_test, predictedGNB)*100, "%\nRecall = ", recallGNBclf, "\n" 'Avg Recall = ', mean(recallGNBclf)*100, "%")

print()

print("\n--Multinomial Naive Bayes--")
print("F1-score = ", f1_score(y_test, predictedMNB, average='macro')*100, "%\n" 'Accuracy =', accuracy_score(y_test, predictedMNB)*100, "%\nRecall = ", recallMNBclf, "\n" 'Avg Recall = ', mean(recallMNBclf)*100, "%")

print()

print("\n--Complement Naive Bayes--")
print("F1-score = ", f1_score(y_test, predictedCNB, average='macro')*100, "%\n" 'Accuracy =', accuracy_score(y_test, predictedCNB)*100, "%\nRecall = ", recallCNBclf, "\n" 'Avg Recall = ', mean(recallCNBclf)*100, "%")

print()

print("\n--Bernoulli Naive Bayes--")
print("F1-score = ", f1_score(y_test, predictedBNB, average='macro')*100, "%\n" 'Accuracy =', accuracy_score(y_test, predictedBNB)*100,  "%\nRecall = ", recallBNBclf, "\n" 'Avg Recall = ', mean(recallBNBclf)*100, "%")

print()

print("\n--Categorical Naive Bayes--")
print("F1-score = ", f1_score(y_test, predictedCatNB, average='macro')*100, "%\n" 'Accuracy =', accuracy_score(y_test, predictedCatNB)*100, "%\nRecall = ", recallCatNBclf, "\n" 'Avg Recall = ', mean(recallCatNBclf)*100, "%")

print()