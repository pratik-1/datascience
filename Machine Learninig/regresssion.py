from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt




# prepare data
boston = datasets.load_boston()
X = boston['data']
y = boston['target']
print('feature_names: ', boston['feature_names'].reshape(1,-1),'\n\n')
df = pd.DataFrame(boston['data'],columns = boston.feature_names)
df['MEDV'] = boston['target']
print(df.describe(),'\n\n')
# print('y',y)
y = y.reshape(-1,1)
# print('y after reshape (-1,1)',y)


# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# build model
print('Simple linear model')
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
# model performance
print('linear regression model performance: ',reg.score(X_test, y_test))

# cross_val_score
cv_results = cross_val_score(reg, X,y,cv=5)
print('5 fold Cross validations: ',np.around(cv_results,3).tolist())

new_data = np.array([5,0.,18,0.,0.71,6,89,2,24.,666.,20,393,10]).reshape(1,-1)
print('  New data:', new_data)
y_new_data = reg.predict(new_data)
print('  Predicted house price on new data: ', y_new_data.item(),'\n\n')





print('Regularised Regression:')
print('      Ridge Regression:')
ridge = Ridge(alpha=0.5, normalize=True)
ridge.fit(X_train, y_train)
ridge_y_pred = ridge.predict(X_test)
print('Ridge regression model performance: ', ridge.score(X_test,y_test))
# cross_val_score
cv_results = cross_val_score(ridge, X,y,cv=5)
print('5 fold Cross validations scores : ',np.around(cv_results,3).tolist())

y_new_data = ridge.predict(new_data)
print('  Predicted house price on new data: ', y_new_data.item(),'\n\n')



print('Regularised Regression:')
print('      Lasso Regression:')
lasso = Lasso(alpha=0.1, normalize=True)
lasso.fit(X_train,y_train)
lasso_y_pred = lasso.predict(X_test)
print('Lasso regression model performance: ', lasso.score(X_test,y_test))
# cross_val_score
cv_results = cross_val_score(lasso, X,y,cv=5)
print('5 fold Cross validations scores : ',np.around(cv_results,3).tolist())

y_new_data = lasso.predict(new_data)
print('  Predicted house price on new data: ', y_new_data.item(),'\n\n')


print('Regularised Regression:')
print('Lasso Regression for feature selection: PLOT')

names = boston['feature_names']
lasso_feature = Lasso(alpha=0.1)
coef = lasso_feature.fit(X_train, y_train).coef_
_ = plt.plot(range(len(names)), coef)
_ = plt.xticks(range(len(names)), names, rotation=60)
_ = plt.ylabel('Correlation Coefficients')
plt.show()


# Data plot
plt.scatter(df['RM'], df['MEDV'])
plt.xlabel('Number of Rooms')
plt.ylabel('House Price $(in Thousands)')
plt.show()



