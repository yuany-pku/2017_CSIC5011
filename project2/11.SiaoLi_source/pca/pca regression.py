# -*- coding: utf-8 -*-
"""
Created on Wed Nov 08 12:28:42 2017

@author: Thinkpad
"""

from sklearn.decomposition import PCA
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

dataset = pd.read_csv('animal dreams.txt',delim_whitespace=True,header =None)
dnp = dataset.drop(0,axis = 1).values
Y = dnp[:,-1]
X = dnp[:,:-1]


pca=PCA(n_components=9)
pca.fit(X)
pca.transform(X)



diabetes_y_train = (Y[:-30]).reshape(-1,1)
diabetes_y_test = (Y[-30:]).reshape(-1,1)

diabetes_X_train = (pca.transform(X))[:-30]
diabetes_X_test = (pca.transform(X))[-30:]

regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
diabetes_y_pred = regr.predict(diabetes_X_test)

print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))


aa = [1.78,1.81,1.44,1.66,1.47,1.39,0.23,0.23,0.26]
bb = [0.00,0.02,0.19,0.07,0.17,0.22,0.87,0.87,0.85]
cc = [1,2,3,4,5,6,7,8,9]
plt.figure(figsize=(20, 6))
plt.subplot(121)
ax = plt.gca()

ax.plot(cc, bb)

ax.set_xscale('log')
plt.xlabel('number of principal components')
plt.ylabel('R^2 ')
plt.title('PCA regression')
plt.axis('tight')

plt.savefig('ab.jpg')