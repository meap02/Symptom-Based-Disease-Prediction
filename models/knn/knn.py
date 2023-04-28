import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
# -------------------------- Models -------------------------- #
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
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
s_best_accuracy = 0
s_best_weight = ""
s_best_algo = ""
s_best_n = 0
s_best_mode = None

e_best_accuracy = 0
e_best_weight = ""
e_best_algo = ""
e_best_n = 0
e_best_mode = None

for n in range(1, 30):
    for weights in ['uniform', 'distance']:
        for algo in ['auto', 'ball_tree', 'kd_tree', 'brute']:
            print("Attempting with {n} neighbors, {weights} weights and {algo} algo".format(n=n, weights=weights, algo=algo))
            single = KNeighborsClassifier(n_neighbors=n, weights=weights, algorithm=algo)
            ensemble = OneVsOneClassifier(single)
            #single.fit(x_train, y_train)
            ensemble.fit(x_train, y_train)
            #s_score = single.score(x_test, y_test)
            e_score = ensemble.score(x_test, y_test)
            '''
            if s_score > s_best_accuracy:
                s_best_accuracy = s_score
                s_best_n = n
                s_best_weight = weights
                s_best_algo = algo
                s_best_model = single
                print("New best accuracy: {accuracy} with {n} neighbors, {weights} weights and {algo} algo".format(accuracy=s_best_accuracy, n=s_best_n, weights=weights, algo=algo))
            '''
            if e_score > e_best_accuracy:
                e_best_accuracy = e_score
                e_best_n = n
                e_best_weight = weights
                e_best_algo = algo
                e_best_model = ensemble
                print("New best accuracy: {accuracy} with {n} neighbors, {weights} weights and {algo} algo".format(accuracy=e_best_accuracy, n=e_best_n, weights=weights, algo=algo))
'''
with open('models/knn/knn.pkl', 'wb') as f:
    pickle.dump(s_best_model, f)
with open('models/knn/knn.json', 'w') as f:
    json.dump({
        'accuracy': s_best_accuracy,
        'n': s_best_n,
        'weights': s_best_weight,
        'algo': s_best_algo
    }, f)
'''
with open('models/knn/knn_oneVone.pkl', 'wb') as f:
    pickle.dump(e_best_model, f)
with open('models/knn/knn_oneVone.json', 'w') as f:
    json.dump({
        'accuracy': e_best_accuracy,
        'n': e_best_n,
        'weights': e_best_weight,
        'algo': e_best_algo
    }, f)
    

    
'''
# -------------------------- Plotting -------------------------- #
'''
