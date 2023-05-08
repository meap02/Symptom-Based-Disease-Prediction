import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
# -------------------------- Models -------------------------- #
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
# -------------------------- Preprocessing -------------------------- #
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, TargetEncoder
# -------------------------- Metrics -------------------------- #
from sklearn.model_selection import train_test_split, cross_val_score



# -------------------------- Data -------------------------- #
df = pd.read_csv('data/patil/dataset.csv').sample(frac=1)
x = df.drop("Disease", axis=1)
y = df["Disease"]
#print(dataframe.drop('Disease', axis=1))
le = LabelEncoder()
te = TargetEncoder(target_type='binary', cv=6)
y = le.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.4, train_size=.6)
x_train = te.fit_transform(x_train, y_train)
x_test = te.transform(x_test)


e_best_accuracy = 0
e_best_c = 0
e_best_kernel = ""

s_best_accuracy = 0
s_best_c = 0
s_best_kernel = ""

try:
    with open('models/svm/svm_onevrest.json', 'r') as f:
        data = json.load(f)
        print("Ensemble model loaded with accuracy {accuracy} and {c} C and {kernel} kernel".format(accuracy=data['accuracy'], c=data['c'], kernel=data['kernel']))
        e_best_accuracy = data['accuracy']
        e_best_c = data['c']
        e_best_kernel = data['kernel']
except Exception as e:
    print(e)
    
try:
    with open('models/svm/svm.json', 'r') as f:
        data = json.load(f)
        print("Single model loaded with accuracy {accuracy} and {c} C and {kernel} kernel".format(accuracy=data['accuracy'], c=data['c'], kernel=data['kernel']))
        s_best_accuracy = data['accuracy']
        s_best_c = data['c']
        s_best_kernel = data['kernel']
except Exception as e:
    print(e)
# -------------------------- Models -------------------------- #
for c in range(17, 30):
    for kernel in ['poly', 'rbf']:
        print("Attemting {c} C and {kernel} kernel".format(c=c, kernel=kernel))
        single = SVC(C = float(c), kernel=kernel, probability=False)
        ensemble = OneVsRestClassifier(single)
        ensemble.fit(x_train, y_train)
        single.fit(x_train, y_train)
        e_evaluation = ensemble.score(x_test, y_test)
        s_evaluation = single.score(x_test, y_test)
        if e_evaluation > e_best_accuracy:
            print("New highest ensemble: {highest} with {c} C and {kernel} kernel".format(highest=e_evaluation, c=c, kernel=kernel))
            e_best_accuracy = e_evaluation
            e_best_c = c
            e_best_kernel = kernel
        if s_evaluation > s_best_accuracy:
            print("New highest single: {highest} with {c} C and {kernel} kernel".format(highest=s_evaluation, c=c, kernel=kernel))
            s_best_accuracy = s_evaluation
            s_best_c = c
            s_best_kernel = kernel

with open('models/svm/svm.pkl', 'wb') as f:
    pickle.dump(single, f)
with open('models/svm/svm.json', 'w') as f:
    json.dump({"accuracy":s_best_accuracy, "c": s_best_c, "kernel": s_best_kernel}, f)
with open('models/svm/svm_onevrest.pkl', 'wb') as f:
    pickle.dump(ensemble, f)
with open('models/svm/svm_onevrest.json', 'w') as f:
    json.dump({"accuracy":e_best_accuracy, "c": e_best_c, "kernel": e_best_kernel}, f)
        


'''
# -------------------------- Plotting -------------------------- #
'''
