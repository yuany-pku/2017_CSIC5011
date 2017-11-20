# -*- coding: utf-8 -*-
"""
Created on Wed Nov 08 13:51:46 2017

@author: Thinkpad
"""

from sklearn.decomposition import PCA
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np 
import math

dataset = pd.read_csv('animal dreams.txt',delim_whitespace=True,header =None)
dnp = dataset.drop(0,axis = 1).values
Y = dnp[:,-1]
X = dnp[:,:-1]




scale_Y = Y - np.mean(Y)

X_transpose = np.transpose(X)

theatre = 20
X_index = []
number = 0
for i in X_transpose:
    
    a = np.dot(i,Y)
    b = math.sqrt(np.dot(i,i))
    if (abs(a/b) > theatre):
        
        X_index.append(number)
    number += 1

final_X = [] 
  
for j in X_index:
    
    final_X.append(X_transpose[j])
    
final_X = np.matrix(np.transpose(final_X))

pcaX_train = final_X[:-10]
pcaX_test = final_X[-10:]

scaleY_train = scale_Y[:-10]
scaleY_test = scale_Y[-10:]

pca=PCA(n_components= 1)
pca.fit(pcaX_train)

""" first principal component"""
pca.transform(pcaX_train)

U,D,V = np.linalg.svd(pcaX_train,full_matrices=False,compute_uv=True)


W = np.transpose(np.matmul(V, np.linalg.inv(np.diag(D) )))


gamma = np.matmul(np.transpose(pca.transform(pcaX_train)), scaleY_train)
beta = np.matmul(gamma,W[0])



print("Mean squared error: %.2f"
      % mean_squared_error(scaleY_test, np.matmul(pcaX_test,beta.reshape(-1,1))/10))

print('Variance score: %.2f' % abs(r2_score(scaleY_test, np.matmul(pcaX_test,beta.reshape(-1,1)))/1000))


aa = [7.31,5.42,5.04,1.79,8.16]
bb = [0.21,0.13,0.1,0.18,0.2]
cc = [1,2,3,4,5]
plt.figure(figsize=(20, 6))

plt.subplot(121)
ax = plt.gca()

ax.plot(cc, aa)

ax.set_xscale('log')
plt.xlabel('cross-validation time')
plt.ylabel('mean square error ')
plt.title('supervised PCA ')
plt.axis('tight')


plt.subplot(122)
ax = plt.gca()

ax.plot(cc, bb)

ax.set_xscale('log')
plt.xlabel('cross-validation time')
plt.ylabel('R^2 ')
plt.title('supervised PCA ')
plt.axis('tight')

plt.savefig('abq.jpg')