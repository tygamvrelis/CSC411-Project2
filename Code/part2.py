## part2.py
# In this file, helper functions are defined for computing a simple network

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

def softmax(y):
    '''
    Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases
    '''
    
    return exp(y)/tile(sum(np.exp(y),0), (len(y),1))
    
def SimpleNetwork(W, X):
    '''
    SimpleNetwork returns the vectorized multiplication of the (n x 10) 
    parameter matrix W with the data X.
    
    Arguments:
        W -- (n x 10) matrix of parameters (weights and biases)
        x -- (n x m) matrix whose i-th column corresponds to the i-th training 
             image
    '''
    
    return softmax(np.dot(W.T, X))