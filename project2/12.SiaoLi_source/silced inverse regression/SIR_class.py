# -*- coding: utf-8 -*-
import numpy as np
from numpy import linalg

# Author: Russell Kunes
# Implementing the SIR supervised data reduction method as described in the 1991 paper by Professor Ker-Chau Li


class SIR:
    def __init__(self, K = 2, H = 10, bins = None):
        self.K = K
        self.H = H
        #default is equally spaced bins
        self.bins = bins

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

        #n is the number of observations
        n = X.shape[0]
        p = X.shape[1]

        x_bar = np.mean(X,axis =0)


        #compute the bins, assuming default 
        if self.bins == None:
        	n_h, bins = np.histogram(Y,bins = self.H)
        else: 
        	n_h,bins = np.histogram(Y, bins = self.bins)

        #assign a bin to each observations
        assignments = np.digitize(Y,bins)

        #this is really hacky... 
        assignments[np.argmax(assignments)] -= 1

        #loop through the slices, for each slice compute within slice mean
        M = np.zeros((p,p))
        for i in range(len(n_h)):

        	h = n_h[i]
        	if h != 0:
        		x_h_bar = np.mean(X[assignments == i + 1],axis = 0)
        	elif h ==0:
        		x_h_bar = np.zeros(p)

        	x_std = x_h_bar - x_bar

        	M += float(h) * np.outer(x_std,x_std)

        #compute the estimate of the covariance matrix M
        M  = float(n)**(-1) * M
        self.M = M

        #eigendecomposition of V
        cov = np.cov(X.T)
        V = np.dot(linalg.inv(cov),M)
        eigenvalues, eigenvectors = linalg.eig(V)

        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]

        #assign first K columns to beta 
        beta = eigenvectors[:,0:self.K]
        self.beta = beta
        return

   	def fit_transform(self,X,Y):
   		#call fit
   		self.fit(X,Y)

   		#get the betas
   		beta = self.beta
   		return np.dot(X,beta)


    def transform(self, X_to_predict):
    	beta = self.beta 
    	return np.dot(X_to_predict,beta)

   	#TODO: 
   	# implement chi square test for elliptical symmetry
   	# method to estimate K 