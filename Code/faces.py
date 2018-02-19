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

## Part 8: Using PyTorch to train a single-hidden-layer fully-connected NN to classify faces
act = ['bracco', 'gilpin', 'harmon', 'baldwin', 'hader', 'carell']
#Note: The act array determines the index of each actor in the one-hot encoding.
RESOLUTION = 32
train_size = 70
val_size = 20
test_size = 20
#improc.make3Sets(RESOLUTION, train_size, val_size, test_size)
(trainX, trainY) = p8.get_set("../Data/Faces/training set" + str(RESOLUTION), RESOLUTION, act)
