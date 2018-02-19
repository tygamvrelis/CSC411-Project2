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

import part8 as p8

## Part 8: Using PyTorch to train a single-hidden-layer fully-connected NN to classify faces
RESOLUTION = 32
train_size = 70
val_size = 20
test_size = 20

p8.preProcess(RESOLUTION, train_size, val_size, test_size)