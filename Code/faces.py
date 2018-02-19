## Faces.py
# This file contains the code to run parts 8 through 10 for CSC411 project 2.
# This involves running machine learning algorithms on faces from the FaceScrub dataset.

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import cPickle
import os
from scipy.io import loadmat
import image_processing as improc
import part8 as p8
import torch
from torch.autograd import Variable

## Part 8: Using PyTorch to train a single-hidden-layer fully-connected NN to classify faces
act = ['bracco', 'gilpin', 'harmon', 'baldwin', 'hader', 'carell']
#Note: The act array determines the index of each actor in the one-hot encoding.
RESOLUTION = 32
train_size = 70
val_size = 20
test_size = 20
#improc.make3Sets(RESOLUTION, train_size, val_size, test_size)
(trainX, trainY) = p8.get_set("../Data/Faces/training set" + str(RESOLUTION), RESOLUTION, act)
(valX, valY) = p8.get_set("../Data/Faces/validation set" + str(RESOLUTION), RESOLUTION, act)

dim_x = RESOLUTION**2*3
dim_h = 500
dim_out = 6

dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor

model = torch.nn.Sequential(
    torch.nn.Linear(dim_x, dim_h),
    torch.nn.Tanh(),
    torch.nn.Linear(dim_h, dim_out),
).cuda()


loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
np.random.seed(0)
for t in range(1000):
    train_idx = np.random.permutation(range(trainX.shape[0]))[:100]
    x = Variable(torch.from_numpy(trainX[train_idx]), requires_grad=False).type(dtype_float).cuda()
    y_classes = Variable(torch.from_numpy(np.argmax(trainY[train_idx], 1)), requires_grad=False).type(dtype_long).cuda()

    y_pred = model(x).cuda()
    loss = loss_fn(y_pred, y_classes)

    model.zero_grad()  # Zero out the previous gradient computation
    loss.backward()    # Compute the gradient
    optimizer.step()   # Use the gradient information to
                       # make a step
x = Variable(torch.from_numpy(valX), requires_grad=False).type(dtype_float).cuda()
y_pred = model(x).cpu().data.numpy()

print np.mean(np.argmax(y_pred, 1) == np.argmax(valY, 1))
# model[0].weight
# model[0].weight.data.numpy()[10, :].shape
plt.imshow(model[0].weight.cpu().data.numpy()[19, 0:RESOLUTION*RESOLUTION].reshape((RESOLUTION, RESOLUTION)), cmap=plt.cm.coolwarm)
plt.show()
plt.imshow(model[0].weight.cpu().data.numpy()[12, 0:RESOLUTION*RESOLUTION].reshape((RESOLUTION, RESOLUTION)), cmap=plt.cm.coolwarm)
plt.show()