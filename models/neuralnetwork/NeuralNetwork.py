import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import time




print("Starting sweep at " + time.strftime("%m-%d--%H:%M:%S:", time.gmtime()))
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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.95, train_size=.05)

best_accuracy = 0
best_activation = None
best_solver = None
best_learning_rate = None
best_model = None
'''
try:
    with open('models/neuralnetwork/neuralnetwork.json', 'r') as f:
        data = json.load(f)
        print("Best neural network loaded with accuracy {accuracy} and {activation} activation, {solver} solver and {learning_rate} learning rate".format(accuracy=data['accuracy'], activation=data['activation'], solver=data['solver'], learning_rate=data['learning_rate']))
        best_accuracy = data['accuracy']
        best_activation = data['activation']
        best_solver = data['solver']
        best_learning_rate = data['learning_rate']
    with open('models/neuralnetwork/neuralnetwork.pkl', 'rb') as f:
        best_model = pickle.load(f)
except Exception as e:
    print(e)
'''
for activation in ['identity', 'logistic', 'tanh', 'relu']:
    for solver in ['lbfgs', 'sgd', 'adam']:
        for learning_rate in ['constant', 'invscaling', 'adaptive']:
            print(time.strftime("%H:%M:%S", time.gmtime()) + " -- Activation: {activation}, Solver: {solver}, Learning rate: {learning_rate} ".format(activation=activation, solver=solver, learning_rate=learning_rate))
            clf = MLPClassifier(activation=activation, solver=solver, learning_rate=learning_rate, max_iter=1000)
            clf.fit(x_train, y_train)
            score = clf.score(x_test, y_test)
            if score > best_accuracy:
                best_accuracy = score
                best_activation = activation
                best_solver = solver
                best_learning_rate = learning_rate
                best_model = clf
                print("New highest: {highest} with {activation} activation, {solver} solver and {learning_rate} learning rate".format(highest=best_accuracy, activation=activation, solver=solver, learning_rate=learning_rate))

'''
with open('models/neuralnetwork/neuralnetwork.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('models/neuralnetwork/neuralnetwork.json', 'w') as f:
    json.dump({"accuracy": best_accuracy, "activation": best_activation, "solver": best_solver, "learning_rate": best_learning_rate}, f)
    '''