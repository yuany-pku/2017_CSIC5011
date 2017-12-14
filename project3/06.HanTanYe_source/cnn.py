#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 19:49:50 2017

@author: Ruijian
"""
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt

torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1     # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 2
LR = 0.0005
  
        
#Raphael
#minus = 200
#previous = 1460
size = 2700
#size_2 = 2200-minus
size_2 = 2200
data_raw_1 = np.load("raph256.npy",encoding='bytes')
d =  torch.FloatTensor(size+size_2,256, 256).zero_()
for i in range(size):
    d[i] = torchvision.transforms.ToTensor()(np.reshape(data_raw_1[i],(256,256,1)))
#for i in range(previous,size):
  #  d[i] = torch.from_numpy(data_raw_1[i+minus]).float()

target_1 = torch.LongTensor(size).zero_()+1
                           
data_raw_2 = np.load("nonraph256.npy",encoding='bytes')
n = torch.FloatTensor(size+size_2,256, 256).zero_()                   
for j in range(size_2):
    d[j+size] = torchvision.transforms.ToTensor()(np.reshape(data_raw_1[j],(256,256,1)))                  
d = torch.unsqueeze(d,dim=1)
target_2 = torch.LongTensor(size_2).zero_()                    
target = torch.cat((target_1,target_2),0)
                           
data_raw_3 = np.load("maybe256.npy",encoding='bytes')

#t_1 = 0
#t_2 = minus
t = 1312
p = 0
tx =  torch.FloatTensor(t, 256, 256).zero_()
#for i in range(t_1):
 #  tx[i] = torch.from_numpy(data_raw_1[i].float() 
for j in range(t):
    tx[j] = torchvision.transforms.ToTensor()(np.reshape(data_raw_1[j+p],(256,256,1)))
#ty_1 = torch.LongTensor(t_1).zero_()+1
#ty_2 = torch.LongTensor(t_2).zero_()
#ty = torch.cat((ty_1,ty_2),0)                  
# Nonraphael




train_dataset = Data.TensorDataset(data_tensor=d, target_tensor=target)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_x = Variable(torch.unsqueeze(tx, dim=1))
#test_y = Variable(ty)

# Mnist 手写数字
#x = torch.from_numpy(raphael_D_3[1]).float()
#y_1 = np.array([0,1])
#y = torch.from_numpy(y_1).float()
#train_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)


#test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
#train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 为了节约时间, 我们测试时只测试前2000个
#test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
#test_y = test_data.test_labels[:2000]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,      # input height
                out_channels=8,    # n_filters
                kernel_size=5,      # filter size
                stride=1,           # filter movement/step
                padding=2,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),      # output shape (16, 28, 28)
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1),# activation
            nn.MaxPool2d(kernel_size=4),    # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(8, 16, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),# activation
            nn.MaxPool2d(4),  # output shape (32, 7, 7)
        )
        self.out_1 = nn.Linear(16 * 16* 16, 16*4)
        self.out_2 =nn.Linear(64,2)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.out_1(x)
        output = F.tanh(output)
        output = self.out_2(output)
        #output = self.out_2(output)
        return output

cnn = CNN()
print(cnn) 
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    print(epoch)
    for step, (x, y) in enumerate(train_loader):   # 分配 batch data, normalize x when iterate train_loader
        b_x = Variable(x)   # batch x
        b_y = Variable(y)   # batch y

        output = cnn(b_x)               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
    
        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0])
            print(torch.mean(pred_y.float()))

