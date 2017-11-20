# -*- coding: utf-8 -*-
"""
Created on Tue Nov 07 13:22:22 2017

@author: Thinkpad
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from SIR_class import SIR
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib inline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets, linear_model

a = np.linspace(1,5,num =5)

dataset = pd.read_csv('animal dreams.txt',delim_whitespace=True,header =None)
dnp = dataset.drop(0,axis = 1).values
Y = dnp[:,-1]
X = dnp[:,:-1]

sir_ = SIR()
sir_.fit(X,Y)

transformed_vals = np.real(sir_.transform(X))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(transformed_vals[:,0],transformed_vals[:,1],Y)
plt.savefig('sir.jpg')
"""
print(transformed_vals[:,0])
print(transformed_vals[:,1])
print(len(transformed_vals[:,0]))
print(len(transformed_vals[:,1]))
"""
######

# Split the data into training/testing sets
diabetes_X_train = ((transformed_vals[:,0])[:-20]).reshape(-1,1)
diabetes_X_test = ((transformed_vals[:,0])[-20:]).reshape(-1,1)


diabetes_Z_train = ((transformed_vals[:,1])[:-20]).reshape(-1,1)
diabetes_Z_test = ((transformed_vals[:,1])[-20:]).reshape(-1,1)



cc = []
for i in range(len((transformed_vals[:,0])[:-20])):
    
    aa = []
    aa.append((transformed_vals[:,0])[:-20][i])
    aa.append((transformed_vals[:,1])[:-20][i])
    cc.append(aa)
 
bb = []   
for j in range(len((transformed_vals[:,0])[-20:])):
    
    aa = []
    aa.append((transformed_vals[:,0])[-20:][j])
    aa.append((transformed_vals[:,1])[-20:][j])
    bb.append(aa)   

"""
print(diabetes_X_train, diabetes_Z_train).reshape(-2,1)
"""
"""
abc = [(transformed_vals[:,0])[:-20],(transformed_vals[:,1])[:-20]].reshape(-1,2)
print(abc)
"""


# Split the targets into training/testing sets
diabetes_y_train = (Y[:-20]).reshape(-1,1)
diabetes_y_test = (Y[-20:]).reshape(-1,1)






# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(cc,diabetes_y_train)
# Make predictions using the testing set
diabetes_y_pred = regr.predict(bb)







print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

##################


x_1 = []
x_2 = []
epsilon = []

for i in xrange(10000):
    epsilon.append(np.random.normal())
    x_1.append(np.random.uniform(0,10))
    x_2.append(np.random.uniform(0,10))
    
x_1 = np.array(x_1)
x_2 = np.array(x_2)
X = zip(x_1,x_2)
X = np.array(X)
epsilon = np.array(epsilon)
y = 2*x_1 + epsilon


"""
plt.figure(figsize = (10,5))
plt.subplot(121,)
plt.xlabel("x_1")
plt.ylabel("y")
plt.scatter(X[:,0],y)
plt.subplot(122)
plt.scatter(X[:,1],y)
plt.xlabel("x_2")
plt.ylabel("y")
plt.savefig('sir1.jpg')
"""
sir_ = SIR(K=1)
sir_.fit(X,y)

sir_.beta

b = [0.37,0.33,0.26,0.29,0.27]
c = [0.82,0.83,0.85,0.85,0.84]



plt.figure(figsize=(20, 6))
plt.subplot(121)
ax = plt.gca()

ax.plot(a, c)

ax.set_xscale('log')
plt.xlabel('cross-validation')
plt.ylabel('R^2 ')
plt.title('sliced inverse regression')
plt.axis('tight')

plt.savefig('fff.jpg')