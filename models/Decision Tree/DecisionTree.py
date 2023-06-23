import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import pathlib
# -------------------------- Models -------------------------- #
from sklearn import tree
# -------------------------- Preprocessing -------------------------- #
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
# -------------------------- Metrics -------------------------- #
from sklearn.model_selection import train_test_split, cross_val_score

cd = pathlib.Path(__file__).parent.absolute()

# -------------------------- Data -------------------------- #
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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.4, train_size=.6)

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.16, train_size=0.84, random_state=42)
# -------------------------- Models -------------------------- #
clf = tree.DecisionTreeClassifier(criterion='entropy')
score = cross_val_score(clf, x, y, cv=6).mean()
clf.fit(x_train, y_train)
for compare in zip(clf.predict(x_test), y_test):
    print(le.inverse_transform(compare))
print(score)

'''
with open(cd / 'decisionTree_.pkl', 'rb') as f:
    clf = pickle.load(f)
print(clf.score(x_test, y_test))
# -------------------------- Plotting -------------------------- #
plt.figure()
tree.plot_tree(clf, feature_names=oe.get_feature_names_out() , class_names=le.classes_, filled=True, rounded=True, fontsize=5,)
plt.show()
'''
# -------------------------- Saving -------------------------- #
#pickle.dump(clf, open('models/decisionTree_.pkl', 'wb'))