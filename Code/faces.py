## Faces.py
# This file contains the code to run parts 8 through 10 for CSC411 project 2.
# This involves running machine learning algorithms on faces from the FaceScrub dataset.

from pylab import *
import numpy as np
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


np.random.seed(0)
## Part 8: Using PyTorch to train a single-hidden-layer fully-connected NN to classify faces
act = ['bracco', 'gilpin', 'harmon', 'baldwin', 'hader', 'carell']
#Note: The act array determines the index of each actor in the one-hot encoding.

dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor
RESOLUTION = 28
train_size = 70
val_size = 20
test_size = 20

#improc.make3Sets(RESOLUTION, train_size, val_size, test_size)

(trainX, trainY) = p8.get_set("../Data/Faces/training set" + str(RESOLUTION), RESOLUTION, act)
(valX, valY) = p8.get_set("../Data/Faces/validation set" + str(RESOLUTION), RESOLUTION, act)
(testX, testY) = p8.get_set("../Data/Faces/validation set" + str(RESOLUTION), RESOLUTION, act)

dim_x = RESOLUTION ** 2 * 3
dim_h = 50
dim_out = 6
model = torch.nn.Sequential(
    torch.nn.Linear(dim_x, dim_h),
    torch.nn.Tanh(),
    torch.nn.Linear(dim_h, dim_out),
).cuda()
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
steps = 2000
batch_size = 50

(tloss_hist,tperf_hist, vloss_hist, vperf_hist, num_iter, model) = p8.train_set(trainX, trainY, valX, valY, model, steps, batch_size, loss_fn, optimizer)


(loss, perf) = p8.classify(valX, valY, model, loss_fn)
print perf
# model[0].weight
# model[0].weight.data.numpy()[10, :].shape
# plt.imshow(model[0].weight.cpu().data.numpy()[5, 0:RESOLUTION*RESOLUTION].reshape((RESOLUTION, RESOLUTION)), cmap=plt.cm.coolwarm)
# plt.show()
# plt.imshow(model[0].weight.cpu().data.numpy()[2, 0:RESOLUTION*RESOLUTION].reshape((RESOLUTION, RESOLUTION)), cmap=plt.cm.coolwarm)
# plt.show()

p8.draw_curves(tloss_hist,tperf_hist, vloss_hist, vperf_hist, num_iter)