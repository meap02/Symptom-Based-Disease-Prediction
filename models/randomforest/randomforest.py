import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
# -------------------------- Models -------------------------- #
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
# -------------------------- Preprocessing -------------------------- #
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, TargetEncoder, OneHotEncoder
# -------------------------- Metrics -------------------------- #
from sklearn.model_selection import train_test_split, cross_val_score


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
#x_train = te.fit_transform(x_train, y_train)
#x_test = te.transform(x_test)
# -------------------------- Models -------------------------- #
best_accuracy = 0
best_rec = 0
best_estimators = 0
best_depth = 0
best_criteria = ""
best_bootstap = False
best_min_samples_split = 0
best_model = None
'''
try:
    with open('models/randomforest/randomforest.json', 'r') as f:
        data = json.load(f)
        best_accuravy = data['accuracy']
        best_estimators = data['estimators']
        best_depth = data['depth']
        best_criteria = data['criteria']
        best_bootstap = data['bootstrap']
    with open('models/randomforest/randomforest.pkl', 'rb') as f:
        best_model = pickle.load(f)
except Exception as e:
    print(e)
'''
plot_acc = []
plot_rec = []
for n in range (1,21):
    for max in range(1, 17):
        for criterion in ['log_loss']:#['entropy', 'log_loss']:
            for bootstap in [False]:#[True, False]:
                print("Attempting with {n} estimators, {max} depth, {criterion} criterion, and {bootstap} bootstap".format(n=n, max=max, criterion=criterion, bootstap=bootstap))
                clf = RandomForestClassifier(n_estimators=n, max_depth=max, criterion='entropy', bootstrap=bootstap)
                acc_score = cross_val_score(clf, x, y, cv=120).mean()
                rec_score = cross_val_score(clf, x, y, cv=120, scoring='recall_micro').mean()
                #clf.fit(x_train, y_train)
                #score = clf.score(x_test, y_test)
                if acc_score > best_accuracy:
                      best_accuracy = acc_score
                      print("New highest: {highest} with {n} estimators and {m} depth".format(highest=best_accuracy, n=n, m=max))
                if rec_score > best_rec: 
                    best_rec = rec_score
                    best_estimators = n
                    best_depth = max
                    best_criteria = criterion
                    best_bootstap = bootstap
                    best_model = clf
                    print("New highest: {highest} with {n} estimators and {m} depth".format(highest=best_rec, n=n, m=max))
    plot_acc.append(best_accuracy)
    plot_rec.append(best_rec)
'''
with open('models/randomforest/randomforest.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('models/randomforest/randomforest.json', 'w') as f:
    json.dump({
        "accuracy": best_accuracy,
        "estimators": best_estimators,
        "depth": best_depth,
        "criteria": best_criteria,
        "bootstrap": best_bootstap
    }, f)
    '''

# -------------------------- Plotting -------------------------- #
plt.plot(range(1,len(plot_acc)+1), plot_acc, '-*', color='red', label='Accuracy')
plt.plot(range(1,len(plot_rec)+1), plot_rec, '-*', color='blue', label='Recall')
plt.title("Random Forest Estimator Count with Optimal Depth vs Recall and Accuracy", fontsize=10)
plt.xlabel("Estimators")
plt.ylabel("Recall and Accuracy Score")
plt.grid()
plt.legend()
plt.savefig('EstimatorvRecallandAccuracy.png')
plt.show()


