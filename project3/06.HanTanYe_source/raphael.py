# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 21:14:51 2017

@author: Roger
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os 
import pandas as pd
import pickle

###load images into 3 lists
imagedir = r'D:\MATH\Data analysis\final_project_2017_12\Raphael_Project_final_copy'
os.chdir(imagedir)
rap = []
os.chdir(imagedir+r'\R')
for r in os.listdir('.'):
    rap.append(mpimg.imread(r))

nonrap = []
os.chdir(imagedir+r'\NR')
for nr in os.listdir('.'):
    nonrap.append(mpimg.imread(nr))
    
mayberap = []
os.chdir(imagedir+r'\maybeR')
for mr in os.listdir('.'):
    mayberap.append(mpimg.imread(mr))

from PIL import Image
from pylab import *
#im = Image.open(os.listdir('.')[0])
imshow(rap[0]) # an array or a PIL image
#%% process image data
##transform to greyscale
def greyshift(image):
    grey = np.zeros(image.shape[0:2])
    grey = image[:,:,0]*0.299 + image[:,:,1]*0.587 + image[:,:,2]*0.114
    return grey
    
#grey1 = greyshift(rap[0])
#plt.imshow(grey1,cmap='Greys_r')

grap = [greyshift(rap[k]) for k in range(0,len(rap))]
gnrap = [greyshift(nonrap[k]) for k in range(0,len(nonrap))]
gmrap = [greyshift(mayberap[k]) for k in range(0,len(mayberap))]
         
##split images into small pieces (ex.256*256)
def splitimage(im, n):
    pieces = [im[x:x+n, y:y+n] for x in range(0,im.shape[0],n) for y in range(0,im.shape[1],n)]
    return pieces

gsrap = [splitimage(grap[l],256) for l in range(0,len(grap))]
gsnrap = [splitimage(gnrap[l],256) for l in range(0,len(gnrap))]
gsmrap = [splitimage(gmrap[l],256) for l in range(0,len(gmrap))]

#combine pieces of different images together and get index of each image
def flattenList(lst):
    result = []
    for sublist in lst:
        result.extend(sublist)
    return result          

def piece(lst,m):
    out = []
    for j in range(len(lst)):
        a = np.floor(lst[j].shape[0]/m)
        b = np.floor(lst[j].shape[1]/m)
        out.append(a*b)
    return out

print(piece(gmrap,256), sum(piece(gmrap,256))) 
mayind = np.cumsum(piece(gmrap,256))-1  #index of images in maybe raphael
rind = np.cumsum(piece(grap,256))-1
nind = np.cumsum(piece(gnrap,256))-1
print(mayind)

trainrap = flattenList(gsrap) 
trainrap1 = [t for t in trainrap if t.shape==trainrap[0].shape]
#np.save(imagedir+'\\raph256',np.array(trainrap1)) 
trainnr = flattenList(gsnrap)
trainnr1 = [t for t in trainnr if t.shape==trainnr[0].shape]
#np.save(imagedir+'\\nonraph256',np.array(trainnr1)) 
maybe = flattenList(gsmrap)
maybe1 = [t for t in maybe if t.shape==maybe[0].shape]
#np.save(imagedir+'\\maybe256',np.array(maybe1))


#pd.DataFrame(trainrap).to_csv(imagedir+'/trainrap.csv',float_format=float)       
#pickle.dump(trainrap,open(imagedir+'/trainrap.txt','wb'), True)
#loadt = pickle.load(open(imagedir+'/trainrap.dat','rb'))
#plt.imshow(sp1[4],cmap='Greys_r')
#pd.DataFrame(gsrap).to_csv(imagedir+'\\raph256.csv')
#pd.DataFrame(gsnrap).to_csv(imagedir+'\\nonrap256.csv')
#pd.DataFrame(gsmrap).to_csv(imagedir+'\\mayberap256.csv')