# -*- coding: utf-8 -*-
"""
Created on Thu Nov 09 13:43:33 2017

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

a = np.linspace(.0, 10.0, num=100)
mse_r = []
r2_r = []

mse_l = []
r2_l = []



for i in a:
    
   clf = Ridge(alpha = i, copy_X=False, fit_intercept=False, max_iter=None,
   normalize=False, random_state=None, solver='auto', tol=0.001)
    
   clf2 = Lasso(alpha= i, copy_X=False, fit_intercept=False, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)
   
   clf.fit(X_train,Y_train)
   clf2.fit(X_train,Y_train)
   
   y1 = clf.predict(X_test)
   y2 = clf2.predict(X_test)
   
   mse_r.append(mean_squared_error(Y_test, y1))
   r2_r.append(r2_score(Y_test, y1))
   
   mse_l.append(mean_squared_error(Y_test, y2))
   r2_l.append(r2_score(Y_test, y2))
   
   
Y_train = Y[:-30]
Y_test = Y[-30:]

X_train = X[:-30]
X_test = X[-30:]

a = np.linspace(.0, 10.0, num=100)
mse_r2 = []
r2_r2 = []

mse_l2 = []
r2_l2 = []



for i in a:
    
   clf = Ridge(alpha = i, copy_X=False, fit_intercept=False, max_iter=None,
   normalize=False, random_state=None, solver='auto', tol=0.001)
    
   clf2 = Lasso(alpha= i, copy_X=False, fit_intercept=False, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)
   
   clf.fit(X_train,Y_train)
   clf2.fit(X_train,Y_train)
   
   y1 = clf.predict(X_test)
   y2 = clf2.predict(X_test)
   
   mse_r2.append(mean_squared_error(Y_test, y1))
   r2_r2.append(r2_score(Y_test, y1))
   
   mse_l2.append(mean_squared_error(Y_test, y2))
   r2_l2.append(r2_score(Y_test, y2))
   
   
   

Y_train = Y[:-40]
Y_test = Y[-40:]

X_train = X[:-40]
X_test = X[-40:]

a = np.linspace(.0, 10.0, num=100)
mse_r3 = []
r2_r3 = []

mse_l3 = []
r2_l3 = []



for i in a:
    
   clf = Ridge(alpha = i, copy_X=False, fit_intercept=False, max_iter=None,
   normalize=False, random_state=None, solver='auto', tol=0.001)
    
   clf2 = Lasso(alpha= i, copy_X=False, fit_intercept=False, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)
   
   clf.fit(X_train,Y_train)
   clf2.fit(X_train,Y_train)
   
   y1 = clf.predict(X_test)
   y2 = clf2.predict(X_test)
   
   mse_r3.append(mean_squared_error(Y_test, y1))
   r2_r3.append(r2_score(Y_test, y1))
   
   mse_l3.append(mean_squared_error(Y_test, y2))
   r2_l3.append(r2_score(Y_test, y2))
   


Y_train = Y[:-10]
Y_test = Y[-10:]

X_train = X[:-10]
X_test = X[-10:]

a = np.linspace(.0, 10.0, num=100)
mse_r4 = []
r2_r4 = []

mse_l4 = []
r2_l4 = []



for i in a:
    
   clf = Ridge(alpha = i, copy_X=False, fit_intercept=False, max_iter=None,
   normalize=False, random_state=None, solver='auto', tol=0.001)
    
   clf2 = Lasso(alpha= i, copy_X=False, fit_intercept=False, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)
   
   clf.fit(X_train,Y_train)
   clf2.fit(X_train,Y_train)
   
   y1 = clf.predict(X_test)
   y2 = clf2.predict(X_test)
   
   mse_r4.append(mean_squared_error(Y_test, y1))
   r2_r4.append(r2_score(Y_test, y1))
   
   mse_l4.append(mean_squared_error(Y_test, y2))
   r2_l4.append(r2_score(Y_test, y2))
   

Y_train = Y[:-50]
Y_test = Y[-50:]

X_train = X[:-50]
X_test = X[-50:]

a = np.linspace(.0, 10.0, num=100)
mse_r5 = []
r2_r5 = []

mse_l5 = []
r2_l5 = []



for i in a:
    
   clf = Ridge(alpha = i, copy_X=False, fit_intercept=False, max_iter=None,
   normalize=False, random_state=None, solver='auto', tol=0.001)
    
   clf2 = Lasso(alpha= i, copy_X=False, fit_intercept=False, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)
   
   clf.fit(X_train,Y_train)
   clf2.fit(X_train,Y_train)
   
   y1 = clf.predict(X_test)
   y2 = clf2.predict(X_test)
   
   mse_r5.append(mean_squared_error(Y_test, y1))
   r2_r5.append(r2_score(Y_test, y1))
   
   mse_l5.append(mean_squared_error(Y_test, y2))
   r2_l5.append(r2_score(Y_test, y2))

plt.figure(figsize=(20, 6))

"""
plt.subplot(121)
ax = plt.gca()
ax.plot(a, mse_r)
ax.plot(a, mse_r2)
ax.plot(a, mse_r3)
ax.plot(a, mse_r4)
ax.plot(a, mse_r5)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('mean square erroe')
plt.title('cross-validation for Ridge')
plt.axis('tight')

plt.subplot(122)
ax = plt.gca()
ax.plot(a, mse_l)
ax.plot(a, mse_l2)
ax.plot(a, mse_l3)
ax.plot(a, mse_l4)
ax.plot(a, mse_l5)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('mean square erroe')
plt.title('cross-validation for Lasso')
plt.axis('tight')


plt.savefig('ccc.jpg')

"""
plt.subplot(121)
ax = plt.gca()
ax.plot(a, r2_r)
ax.plot(a, r2_r2)
ax.plot(a, r2_r3)
ax.plot(a, r2_r4)
ax.plot(a, r2_r5)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.title('cross-validation for Ridge')
plt.axis('tight')

plt.subplot(122)
ax = plt.gca()
ax.plot(a, r2_l)
ax.plot(a, r2_l2)
ax.plot(a, r2_l3)
ax.plot(a, r2_l4)
ax.plot(a, r2_l5)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.title('cross-validation for Lasso')
plt.axis('tight')


plt.savefig('ddd.jpg')
