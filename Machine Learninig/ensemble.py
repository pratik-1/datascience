from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import datasets
import pandas as pd


iris = datasets.load_iris()
X = iris.data
y = iris.target.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

score = rf.score(X_test, y_test)
print('Score', score)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:",accuracy)

new_data = [3, 5, 4, 2]
new_pred = rf.predict(new_data)
print('New data prediction',iris.target_names[new_pred])


### Votting ensemble

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import KFold


# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('dtree', model2))
model3 = SVC()
estimators.append(('svm', model3))

# create the ensemble model
ensemble = VotingClassifier(estimators)
new_pred_ensemble = rf.predict(new_data)
print('New data prediction ensemble',iris.target_names[new_pred_ensemble])

kfold = KFold(n_splits=5, random_state=2)
results = cross_val_score(ensemble, X, y, cv=kfold)
print('Results ensemble', results.mean())

