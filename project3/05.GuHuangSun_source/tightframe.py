import numpy as np
from PIL import Image
from scipy.signal import convolve2d

from os import listdir
from os.path import join
from time import time

filters = [
    np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
    np.array([[1, 0, -1], [2, 0, -2,], [1, 0, -1]]) / 16,
    np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 16,
    np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]]) * np.sqrt(2) / 16,
    np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]]) * np.sqrt(2) / 16,
    np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) * np.sqrt(7) / 24,
    np.array([[-1, 2, -1], [-2, 4, -2], [-1, 2, -1]]) / 48,
    np.array([[-1, -2, -1], [2, 4, 2], [-1, -2, -1]]) / 48,
    np.array([[0, 0, -1], [0, 2, 0], [-1, 0, 0]]) / 12,
    np.array([[-1, 0, 0], [0, 2, 0], [0, 0, -1]]) / 12,
    np.array([[0, 1, 0], [-1, 0, -1], [0, 1, 0]]) * np.sqrt(2) / 12,
    np.array([[-1, 0, 1], [2, 0, -2], [-1, 0, 1]]) * np.sqrt(2) / 16,
    np.array([[-1, 2, -1], [0, 0, 0], [1, -2, 1]]) * np.sqrt(2) / 16,
    np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]]) / 48,
    np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]]) * np.sqrt(2) / 12,
    np.array([[-1, 2, -1], [0, 0, 0], [-1, 2, -1]]) * np.sqrt(2) / 24,
    np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]]) * np.sqrt(2) / 12,
    np.array([[-1, 0, -1], [2, 0, 2], [-1, 0, -1]]) * np.sqrt(2) / 24
]

n_fil = len(filters)
filelist = listdir('pics')
n_files = len(filelist)

# Read images, convert into grayscale, apply filters, and extract features
sortedid = np.argsort([int(j.split('.')[0]) for j in filelist])
sortedlist = [filelist[i] for i in sortedid]
for i,j in enumerate(sortedlist):
    assert (i+1) == int(j.split('.')[0])
print(sortedlist)
n_features = n_fil * 3

features = np.zeros((n_files, n_features))
for i,pic in enumerate(sortedlist):
    print('Processing image: {}'.format(i+1), end='\r')
    start = time()
    im = np.asarray(Image.open(join('pics',pic)).convert('L'))
    mu, std, p = [], [], []
    for j,f in enumerate(filters):
        coeff = convolve2d(im, f, 'same')
        mu.append(coeff.mean())
        std.append(coeff.std(ddof=1))
        p.append(np.greater(np.abs(coeff - mu[j]), std[j]).mean())
    feature = mu + std + p
    features[i,:] = feature
    print('Image {} done. Time used: {}'.format(i+1, time()-start))

np.savetxt('features.txt', features)
