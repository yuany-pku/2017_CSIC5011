#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 22:17:48 2017

@author: Roger
"""

import numpy as np
import pandas as pd
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import os
os.getcwd()
os.chdir(r'D:\MATH\Data analysis\final_project_2017_12')
from raphael import trainrap1,trainnr1,maybe1  #dataset of raphael and non-raphael

#%% hyperparameters
EPOCH = 3
BATCH_SIZE = 10
LR = 0.001

#transform to tensors
imtensor = torch.FloatTensor(len(trainrap1),256,256).zero_()
for k in range(0,len(trainrap1)):
    imtensor[k] = torchvision.transforms.ToTensor()(np.reshape(trainrap1[k],(256,256,1)))
X = torch.unsqueeze(imtensor,1)

#y = np.array([1])
#y = torch.from_numpy(y).float()
train_dataset = Data.TensorDataset(data_tensor=X, target_tensor=X)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#define structure of autoencoder
class AutoEncoder(nn.Module):
    def __init__(self,in_dim,hidden_1,hidden_2):
        super(AutoEncoder, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_1),
            nn.Dropout(0.5),
            nn.Sigmoid(),
            nn.Linear(hidden_1, hidden_2),
            #nn.Dropout(0.5),
            #nn.ReLU(),
            #nn.Linear(hidden_2, hidden_3),   
        )
        # decoder
        self.decoder = nn.Sequential(
           # nn.Linear(hidden_3, hidden_2),
           #          nn.Dropout(0.5),
           # nn.ReLU(),
            nn.Linear(hidden_2, hidden_1),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(hidden_1, in_dim),
          )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = AutoEncoder(256*256, 300, 10)
optimizer = torch.optim.Adam(autoencoder.parameters(),lr=LR,betas=(0.9,0.999),eps = 1e-6,weight_decay =1e-6)
loss_func = nn.MSELoss()
#training autoencoder
for epoch in range(EPOCH):
    running_loss = 0.0
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x.view(-1,256*256))   # batch x, shape (batch, 28*28)
        b_y = Variable(x.view(-1,256*256))   # batch y, shape (batch, 28*28)
        #b_label = Variable(y)               # batch label

        encoded, decoded = autoencoder(b_x)

        loss = loss_func(decoded, b_y)      # mean square error
        running_loss += loss.data[0]        # loss sum
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradientsf
        if step % 100 == 0:
           print('Epoch: ', epoch+1, '| train loss: %.4f' % (running_loss/(step+1)))

#%%use trained autoencoder to extract features
npfea = autoencoder(Variable(X.view(-1,256*256)))[0].data.numpy()
#%%the second autoencoder for non-raphael
EPOCH = 3
BATCH_SIZE = 10
LR = 0.001

#transform to tensors
imtensor = torch.FloatTensor(len(trainnr1),256,256).zero_()
for k in range(0,len(trainnr1)):
    imtensor[k] = torchvision.transforms.ToTensor()(np.reshape(trainnr1[k],(256,256,1)))
X = torch.unsqueeze(imtensor,1)

#y = np.array([1])
#y = torch.from_numpy(y).float()
train_dataset = Data.TensorDataset(data_tensor=X, target_tensor=X)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#define structure of autoencoder
class AutoEncoder(nn.Module):
    def __init__(self,in_dim,hidden_1,hidden_2):
        super(AutoEncoder, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_1),
            nn.Dropout(0.5),
            nn.Sigmoid(),
            nn.Linear(hidden_1, hidden_2),
            #nn.Dropout(0.5),
            #nn.ReLU(),
            #nn.Linear(hidden_2, hidden_3),   
        )
        # decoder
        self.decoder = nn.Sequential(
           # nn.Linear(hidden_3, hidden_2),
           #          nn.Dropout(0.5),
           # nn.ReLU(),
            nn.Linear(hidden_2, hidden_1),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(hidden_1, in_dim),
          )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder2 = AutoEncoder(256*256, 300, 10)
optimizer = torch.optim.Adam(autoencoder2.parameters(),lr=LR,betas=(0.9,0.999),eps = 1e-6,weight_decay =1e-6)
loss_func = nn.MSELoss()
#training autoencoder
for epoch in range(EPOCH):
    running_loss = 0.0
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x.view(-1,256*256))   # batch x, shape (batch, 28*28)
        b_y = Variable(x.view(-1,256*256))   # batch y, shape (batch, 28*28)
        #b_label = Variable(y)               # batch label

        encoded, decoded = autoencoder2(b_x)

        loss = loss_func(decoded, b_y)      # mean square error
        running_loss += loss.data[0]        # loss sum
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradientsf
        if step % 100 == 0:
           print('Epoch: ', epoch+1, '| train loss: %.4f' % (running_loss/(step+1)))

#%%extract features of non-raphael
npnr = autoencoder2(Variable(X.view(-1,256*256)))[0].data.numpy()
#%%the third autoencoder for maybe-raphael
EPOCH = 3
BATCH_SIZE = 10
LR = 0.001

#transform to tensors
imtensor = torch.FloatTensor(len(maybe1),256,256).zero_()
for k in range(0,len(maybe1)):
    imtensor[k] = torchvision.transforms.ToTensor()(np.reshape(maybe1[k],(256,256,1)))
X = torch.unsqueeze(imtensor,1)

#y = np.array([1])
#y = torch.from_numpy(y).float()
train_dataset = Data.TensorDataset(data_tensor=X, target_tensor=X)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#define structure of autoencoder
class AutoEncoder(nn.Module):
    def __init__(self,in_dim,hidden_1,hidden_2):
        super(AutoEncoder, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_1),
            nn.Dropout(0.5),
            nn.Sigmoid(),
            nn.Linear(hidden_1, hidden_2),
            #nn.Dropout(0.5),
            #nn.ReLU(),
            #nn.Linear(hidden_2, hidden_3),   
        )
        # decoder
        self.decoder = nn.Sequential(
           # nn.Linear(hidden_3, hidden_2),
           #          nn.Dropout(0.5),
           # nn.ReLU(),
            nn.Linear(hidden_2, hidden_1),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(hidden_1, in_dim),
          )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder3 = AutoEncoder(256*256, 300, 10)
optimizer = torch.optim.Adam(autoencoder3.parameters(),lr=LR,betas=(0.9,0.999),eps = 1e-6,weight_decay =1e-6)
loss_func = nn.MSELoss()
#training autoencoder
for epoch in range(EPOCH):
    running_loss = 0.0
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x.view(-1,256*256))   # batch x, shape (batch, 28*28)
        b_y = Variable(x.view(-1,256*256))   # batch y, shape (batch, 28*28)
        #b_label = Variable(y)               # batch label

        encoded, decoded = autoencoder3(b_x)

        loss = loss_func(decoded, b_y)      # mean square error
        running_loss += loss.data[0]        # loss sum
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradientsf
        if step % 100 == 0:
           print('Epoch: ', epoch+1, '| train loss: %.4f' % (running_loss/(step+1)))

#%%extract features of non-raphael
npmay = autoencoder3(Variable(X.view(-1,256*256)))[0].data.numpy()
#%% logistic regression to predict
from sklearn import linear_model
from sklearn.model_selection import LeaveOneOut

totalfea = np.concatenate((npfea,npnr),axis=0)
target1 = np.ones(npfea.shape[0])
target0 = np.zeros(npnr.shape[0])
totaltarget = np.concatenate((target1,target0),axis=0)
logreg = linear_model.LogisticRegression()
logreg.fit(totalfea,totaltarget)
logreg.coef_

from raphael import mayind, rind, nind
#leve-one-out cross-validation
mayindex = np.insert(mayind,0,0)
mayindex = mayindex.astype(np.int)

rindex = np.insert(rind,0,0)
rindex = rindex.astype(np.int)

nindex = np.insert(nind,0,0)
nindex = nindex.astype(np.int)

def label(fea,index):
    res = []
    for i in range(len(index)-1):
        mm = fea[index[i]:index[i+1]]
        res.append(mm)
    return res
r1 = label(npfea,rindex)
y1 = label(np.ones(npfea.shape[0]),rindex)
r2 = label(npnr, nindex)
y2 = label(np.zeros(npnr.shape[0]),nindex)
r3 = label(npmay,mayindex)

rr = np.asarray(r1+r2) 
ry = np.asarray(y1+y2)

def logcv(xx,yy):
    loo = LeaveOneOut()
    res = []
    for train_ind, test_ind in loo.split(xx):
        logreg = linear_model.LogisticRegression()
        logreg.fit(np.concatenate(xx[train_ind],axis=0),np.hstack(yy[train_ind]))
        mm = np.mean(logreg.predict(np.vstack(xx[test_ind])))
        res.append((mm,np.mean(np.vstack(yy[test_ind]))))
    return res

print(logcv(rr,ry),len(logcv(rr,ry)))

logreg1 = linear_model.LogisticRegression()
logreg1.fit(totalfea,totaltarget)
logreg1.coef_
out = []
for k in range(len(r3)):
    out.append(np.mean(logreg1.predict(r3[k])))
print(out)





