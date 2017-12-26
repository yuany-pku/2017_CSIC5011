import csv
import numpy as np
import sys
import os, fnmatch
import matplotlib.pyplot as plt
from skimage import feature
import cv2
import math
from scipy import signal
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from sklearn.model_selection import KFold

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def readLabelCSV(csvFile):
    csvList = []
    with open(csvFile, 'r') as csvin:
        csvreader = csv.reader(csvin, delimiter=',')
        for line in csvreader:
            floatline = []
            for idx, item in enumerate(line):
                try:
                    floatline.append(int(item))
                except ValueError:
                    a = 0
            csvList.append(floatline)
    return np.array(csvList[0])

imgNumberList = range(1,29)
imgList = []

for i in imgNumberList:
    imgName = find(str(i)+'.*', './')[0]
    img = cv2.imread(imgName)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgList.append(grayImg)


convHistList_of_ImgList = []
label = readLabelCSV('label.csv')
rfilterImgHistFile = find('convHist.csv','./')
if len(rfilterImgHistFile) == 0:

    #                           0th
    raymondFilter = np.array([[[1,2,1],
                               [2,4,2],
                               [1,2,1]],
                              # 1nd
                              [[1,0,-1],
                               [2,0,-2],
                               [1,0,-1]],
                              # 2th
                              [[1,2,1],
                               [0,0,0],
                               [-1,-2,-1]],
                              # 3rd
                              [[1,1,0],
                               [1,0,-1],
                               [0,-1,-1]],
                              # 4th
                              [[0,1,1],
                               [-1,0,1],
                               [-1,-1,0]],
                              # 5th
                              [[1,0,-1],
                               [0,0,0],
                               [-1,-2,-1]],
                              # 6th
                              [[-1,2,-1],
                               [-2,4,-2],
                               [-1,2,-1]],
                              # 7th
                              [[-1,-2,-1],
                               [2,4,2],
                               [-1,-2,-1]],
                              # 8th
                              [[0,0,-1],
                               [0,2,0],
                               [-1,0,0]],
                              # 9th
                              [[-1,0,0],
                               [0,2,0],
                               [0,0,-1]],
                              # 10th
                              [[0,1,0],
                               [-1,0,-1],
                               [0,1,0]],
                              # 11th
                              [[-1,0,1],
                               [2,0,-2],
                               [-1,0,1]],
                              # 12th
                              [[-1,2,-1],
                               [0,0,0],
                               [1,-2,1]],
                              # 13th
                              [[1,-2,1],
                               [-2,4,-2],
                               [1,-2,1]],
                              # 14th
                              [[0,0,0],
                               [-1,2,-1],
                               [0,0,0]],
                              # 15th
                              [[-1,2,-1],
                               [0,0,0],
                               [-1,2,-1]],
                              # 16th
                              [[0,-1,0],
                               [0,2,0],
                               [0,-1,0]],
                              # 17th
                              [[-1,0,-1],
                               [2,0,2],
                               [-1,0,-1]]
                              ]).astype('float')

    print raymondFilter
    raymondFilterFactor = np.array([1.0/16, 1.0/16, 1.0/16, np.sqrt(2)/16, np.sqrt(2)/16, np.sqrt(7)/24,
                           1.0/48, 1.0/48, 1.0/12, 1.0/12, np.sqrt(2)/12, np.sqrt(2)/16,
                           np.sqrt(2)/16, 1.0/48, np.sqrt(2)/12, np.sqrt(2)/24, np.sqrt(2)/12, np.sqrt(2)/24])

    # print raymondFilterFactor
    raymondFilterNormalized = []
    for idx, rfilter in enumerate(raymondFilter):
        raymondFilterNormalized.append(rfilter*raymondFilterFactor[idx])

    raymondFilterNormalized = np.array(raymondFilterNormalized)
    print raymondFilterNormalized.shape

    randomizedFilter = np.array(raymondFilter.shape)

    maxOfImage = 0
    minOfImage = 0

    for idx, grayImg in enumerate(imgList):
        convHistList = np.empty((0))
        print idx
        for filterID, rfilter in enumerate(raymondFilterNormalized):
            convImg = signal.convolve2d(grayImg, rfilter, boundary='symm', mode='same')
            if filterID == 1:
                plt.clf()
                plt.imshow(convImg)
                plt.savefig('test'+str(i)+'.png')
            convImgHist, binRange = np.histogram(convImg.ravel(), bins=(np.array(range(512))-256), density=True)
            if max(convImg.ravel()) > maxOfImage:
                maxOfImage = max(convImg.ravel())
            if min(convImg.ravel()) < minOfImage:
                minOfImage = min(convImg.ravel())
            convHistList = np.append(convHistList, convImgHist)
        print maxOfImage
        print minOfImage
        convHistList_of_ImgList.append(convHistList)

    convHistList_of_ImgList = np.array(convHistList_of_ImgList)

    print convHistList_of_ImgList

    np.savetxt('convHist.csv', convHistList_of_ImgList, delimiter=',')
else:
    convHistList_of_ImgList = np.loadtxt('convHist.csv', delimiter=',')

print convHistList_of_ImgList.shape

kf = KFold(n_splits=10)

def plot_embedding(X, imgFileName, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.clf()
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], labelMeaning[label[i]+1],
                 color=plt.cm.tab10((label[i]+1)/3.),
                 fontdict={'weight': 'bold', 'size': 9})
    #
    # if hasattr(offsetbox, 'AnnotationBbox'):
    #     # only print thumbnails with matplotlib > 1.0
    #     shown_images = np.array([[1., 1.]])  # just something big
    #     for i in range(digits.data.shape[0]):
    #         dist = np.sum((X[i] - shown_images) ** 2, 1)
    #         if np.min(dist) < 4e-3:
    #             # don't show points that are too close
    #             continue
    #         shown_images = np.r_[shown_images, [X[i]]]
    #         imagebox = offsetbox.AnnotationBbox(
    #             offsetbox.OffsetImage(digitImg[i], cmap=plt.cm.gray_r),
    #             X[i])
    #         ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.savefig(imgFileName)


labelMeaning = ["R'", 'R?', 'R!']
n_neighbors = 2
n_components = 3
recall = 0

# xTrain, xTest = convHistList_of_ImgList[train_index], convHistList_of_ImgList[test_index]
# yTrain, yTest = label[train_index], label[test_index]
paintingsne = manifold.TSNE(n_components=2)
xPaintingsne = paintingsne.fit_transform(convHistList_of_ImgList)


isoMap = manifold.Isomap(n_neighbors, n_components)
xIsomap = isoMap.fit_transform(convHistList_of_ImgList)

se = manifold.SpectralEmbedding(n_components=n_components,
                                    n_neighbors=n_neighbors)
xSE = se.fit_transform(convHistList_of_ImgList)


plot_embedding(xPaintingsne, 'tSNE_convHist_paint.png', title='t-SNE of painting with convHist')
plot_embedding(xIsomap, 'isomap_convHist_paint.png', title='ISOMAP of painting with convHist')
plot_embedding(xSE, 'spectralEmbedding_convHist_paint.png', title='Spectral Embedding of painting with convHist')


#
# img1 = cv2.imread('1.tif')
# grayImg = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('gray1.tif', grayImg)








from sklearn.model_selection import cross_val_score
