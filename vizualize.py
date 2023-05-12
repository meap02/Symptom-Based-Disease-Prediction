from sklearn.tree import export_graphviz
import pickle
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data/patil/dataset.csv')#.dropna().sample(frac=1)
x = df.drop("Disease", axis=1)
y = df["Disease"]
#print(dataframe.drop('Disease', axis=1))
le = LabelEncoder()
te = TargetEncoder(target_type='binary', cv=6)
y = le.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)
x_train = te.fit_transform(x_train, y_train)
x_test = te.transform(x_test)


with open("models/randomforest/randomforest.json", 'r') as f:
    params = json.load(f)

clf = RandomForestClassifier(n_estimators=params['estimators'], max_depth=params['depth'], criterion=params['criteria'], bootstrap=params['bootstrap'])
clf.fit(x_train, y_train)
print(clf.score(x_test, y_test))
sample = x.loc[44].to_frame().T
print(sample)
print(te.transform(sample))
print(le.inverse_transform(clf.predict(te.transform(sample))))