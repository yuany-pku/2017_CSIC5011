# -*- coding: utf-8 -*-
__author__ = 'huangyf'

import os
import re
import time
import math
import numpy as np
from skimage import transform
import cv2
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import models, transforms


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since):
    now = time.time()
    s = now - since
    return '%s ' % (asMinutes(s))


class painting(Dataset):
    def __init__(self, root_dir, img_names, labels, transform=None):
        self.root_dir = root_dir
        self.img_names = img_names
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.root_dir, img_name)
        if os.path.exists(img_path+'.npy'):
            image=np.load(img_path+'.npy')
        else:
            image = cv2.imread(img_path)
            np.save(img_path+'.npy',image)
        label = self.labels[idx]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return {'image': img, 'label': label}


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]
        return {'image': image, 'label': label}


class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'label': label}


def leaveOne(X, idx):
    return [X[i] for i in range(len(X)) if i != idx]


def normalize(imgs):
    return transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(imgs)


batch_size = 10
lr = 0.01
num_epochs = 11
num_cv = 21
n_classes = 2
use_cuda = torch.cuda.is_available()
# use_cuda=False

root_dir = '../yuany_course/project3'
names = [name for name in os.listdir(root_dir) if name.lower().endswith(('tif', 'tiff', 'jpg'))]
training_names = [name for name in names if int(re.findall(r'(\d+).', name)[0]) not in [1, 7, 10, 20, 23, 25, 26]]
training_names.sort(key=lambda x: int(re.findall(r'(\d+).', x)[0]))
training_labels = [1] * 7 + [0] * 9 + [1] * 5
testing_names = [name for name in names if name not in training_names]
testing_names.sort(key=lambda x: int(re.findall(r'(\d+).', x)[0]))
print('training samples:{}\ntesting samples {}'.format(len(training_names), len(testing_names)))

testing_datasets = painting(root_dir, testing_names, training_labels[:7],
                            transform=transforms.Compose([Rescale(512), RandomCrop(224),
                                                          ToTensor()]))
testing_loader = DataLoader(testing_datasets, batch_size=len(testing_datasets), shuffle=False)

start = time.time()
cv_result = []
test_predictions = []

for j in range(num_cv):

    print('cv:{} start......'.format(j + 1))

    training_datasets = painting(root_dir, leaveOne(training_names, j), leaveOne(training_labels, j),
                                 transform=transforms.Compose([Rescale(512), RandomCrop(224),
                                                               ToTensor()]))
    training_loader = DataLoader(training_datasets, batch_size=batch_size, shuffle=True)
    val_datasets = painting(root_dir, [training_names[j]], [training_labels[j]],
                            transform=transforms.Compose([Rescale(512), RandomCrop(224),
                                                          ToTensor()]))
    val_loader = DataLoader(val_datasets, batch_size=1, shuffle=False)

    # model = models.resnet18(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = False
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 2)
    model=torch.load('model')
    if use_cuda:
        model = model.cuda()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    accuracy_ = []
    loss_ = []
    num_sample = 0

    for epoch in range(num_epochs):
        for i, data in enumerate(training_loader):
            imgs, labels = data['image'], data['label']
            imgs = normalize(imgs)
            imgs, labels = Variable(imgs), Variable(labels)
            if use_cuda:
                imgs, labels = imgs.cuda(), labels.cuda()
            output = model.forward(imgs)
            output = F.log_softmax(output)
            predictions = output.topk(1)[1].cpu().data.numpy().reshape(-1, )
            accuracy = np.sum(predictions == labels.cpu().data.numpy())
            num_sample += batch_size
            optimizer.zero_grad()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            loss_.append(loss.data[0])
            accuracy_.append(accuracy)
        if epoch % 10 == 0:
            print('time:{} epoch:{} loss:{:.4f},accuracy:{:.4f}'.format(timeSince(start), epoch + 1,
                                                                              sum(loss_) / (
                                                                                  (epoch + 1) * len(
                                                                                      training_datasets) // batch_size),
                                                                              sum(accuracy_) / num_sample))
    model.eval()
    predictions_eval = []
    for j in range(10):
        for data in val_loader:
            imgs, labels = data['image'], data['label']
            imgs = normalize(imgs)
            imgs, labels = Variable(imgs), Variable(labels)
            if use_cuda:
                imgs, labels = imgs.cuda(), labels.cuda()
            output = model.forward(imgs)
            output = F.log_softmax(output)
            predictions = output.topk(1)[1].cpu().data.numpy().reshape(-1, )
            predictions_eval.append(predictions)
    cv_result.append(np.argmax(np.bincount(np.array(predictions_eval).reshape(-1,).astype(np.int32))))
    print('true:{} prediction:{}'.format(data['label'].numpy(), cv_result[-1]))

    predictions_test = []
    for j in range(10):
        for i, data in enumerate(testing_loader):
            imgs, labels = data['image'], data['label']
            imgs = normalize(imgs)
            imgs, labels = Variable(imgs), Variable(labels)
            if use_cuda:
                imgs, labels = imgs.cuda(), labels.cuda()
            output = model.forward(imgs)
            output = F.log_softmax(output)
            predictions = output.topk(1)[1].cpu().data.numpy().reshape(-1, )
            predictions_test.append(predictions)
    predictions_test = np.array(predictions_test)
    predictions = []
    for i in range(7):
        predictions.append(np.argmax(np.bincount(predictions_test[:, i].astype(np.int32))))
    test_predictions.append(predictions)

print('cv score:{}'.format(sum(1 for i in range(21) if cv_result[i]==training_labels[i])/21))