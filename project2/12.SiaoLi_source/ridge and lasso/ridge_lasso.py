# -*- coding: utf-8 -*-
"""
Created on Thu Nov 09 11:10:36 2017

@author: Thinkpad
"""

from sklearn.decomposition import PCA
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np 
import math
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

dataset = pd.read_csv('animal dreams.txt',delim_whitespace=True,header =None)
dnp = dataset.drop(0,axis = 1).values
Y = dnp[:,-1]
X = dnp[:,:-1]


Y_train = Y[:-20]
Y_test = Y[-20:]

X_train = X[:-20]
X_test = X[-20:]



""" 
the value of alpha
"""
b = 0.3

clf = Ridge(alpha = b, copy_X=False, fit_intercept=False, max_iter=None,
   normalize=False, random_state=None, solver='auto', tol=0.001)

clf.fit(X_train,Y_train)

Y_predict = clf.predict(X_test)

print("Mean squared error: %.2f"
      % mean_squared_error(Y_test, Y_predict))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y_test, Y_predict))


regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, Y_train)
"""
# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)
"""
w = regr.coef_

coefs = []
errors = []
mse = []
r2 = []
alphas = np.logspace(-6, 6, 200)

# Train the model with different regularisation strengths
for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(X_train, Y_train)
    coefs.append(clf.coef_)
    Y_predict = clf.predict(X_test)
    errors.append(mean_squared_error(clf.coef_, w))
    mse.append(mean_squared_error(Y_test, Y_predict))
  
    r2.append(r2_score(Y_test, Y_predict))
    
    
    
  
    
    

# Display results
plt.figure(figsize=(20, 6))

plt.subplot(121)
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('Coefficient error as a function of the regularization')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')

plt.subplot(122)
ax = plt.gca()
ax.plot(alphas, errors)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('error')
plt.title('Coefficient error as a function of the regularization')
plt.axis('tight')

plt.savefig('aaa.jpg')


"""
lasso
"""


clf2 = Lasso(alpha= b, copy_X=False, fit_intercept=False, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)

clf2.fit(X_train,Y_train)
Y_predict2 = clf2.predict(X_test)

print("Mean squared error: %.2f"
      % mean_squared_error(Y_test, Y_predict2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y_test, Y_predict2))

mse2 = []
r22 = []

coefs = []
errors = []

alphas = np.logspace(-6, 6, 200)

# Train the model with different regularisation strengths
for a in alphas:
    clf2.set_params(alpha=a)
    clf2.fit(X_train, Y_train)
    Y_predict = clf2.predict(X_test)
    coefs.append(clf2.coef_)
    errors.append(mean_squared_error(clf2.coef_, w))
    mse2.append(mean_squared_error(Y_test, Y_predict))
  
    r22.append(r2_score(Y_test, Y_predict))
    
    
# Display results
plt.figure(figsize=(20, 6))

plt.subplot(121)
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('Coefficient error as a function of the regularization')
plt.title('Lasso coefficients as a function of the regularization')
plt.axis('tight')

plt.subplot(122)
ax = plt.gca()
ax.plot(alphas, errors)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('error')
plt.title('Coefficient error as a function of the regularization')
plt.axis('tight')
"""
plt.savefig('bbb.jpg')

"""
