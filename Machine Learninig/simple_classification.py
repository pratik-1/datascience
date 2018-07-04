from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bokeh.plotting import figure
from bokeh.models import Legend
from bokeh.models import CategoricalColorMapper
from bokeh.io import output_file, show


# load dataset
iris=datasets.load_iris()
X=iris['data']
y=iris['target']

# Initialise Preprocessor/Estimator
steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]

# Create Pipeline
pipeline = Pipeline(steps)
parameters = {'knn__n_neighbors':np.arange(1, 50)}


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=21)
cv = GridSearchCV(pipeline, param_grid=parameters)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)

print(cv.best_params_)
print(cv.best_score_)
print(classification_report(y_test,y_pred))

unseen_data1 = np.array([7, 5,  5, 3.5]).reshape(1,-1)
yhat_pred = cv.predict(unseen_data1)
print(unseen_data1,iris['target_names'][yhat_pred],sep=':')

unseen_data2 = np.array([0.2, 3.5,  0.5, 0.2]).reshape(1,-1)
yhat_pred = cv.predict(unseen_data2)
print(unseen_data2,iris['target_names'][yhat_pred],sep=':')

unseen_data3 = np.array([10.2, 3.5,  1.5, 7.2]).reshape(1,-1)
yhat_pred = cv.predict(unseen_data3)
print(unseen_data3,iris['target_names'][yhat_pred],sep=':')


iris1 = pd.DataFrame(X, columns=iris.feature_names)
iris1['species1'] = iris['target_names'][y]

# print(iris1.head())
# mapper = CategoricalColorMapper(factors=['setosa','virginica','versicolor'], palette=['red', 'green', 'blue'])
# plot = figure()
# plot = figure(x_axis_label='petal length (cm)', y_axis_label='sepal length (cm)')
# plot.circle('petal length (cm)', 'sepal length (cm)', size=10, source=iris1,
#             color={'field': 'species','transform': mapper},
#             legend='species')

mapper = CategoricalColorMapper(factors=['setosa', 'virginica','versicolor'],palette=['red', 'green', 'blue'])
pt = figure(x_axis_label='petal_length',y_axis_label='sepal_length')
pt.circle('petal length (cm)', 'sepal length (cm)',size=10, source=iris1,
            color={'field': 'species1','transform': mapper})

pt.legend.location = 'top_left'
# output_file('iris.html')
show(pt)