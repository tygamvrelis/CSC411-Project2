## part3.py
# In this file, helper functions are defined for computing the gradient of
# the cost function of negative log probabilities

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

import part2 as p2

def NLL(P, Y):
    return -sum(Y*log(P)) 

def negLogLossGrad(X, Y, W):
    '''
    negLogLossGrad returns the gradient of the cost function of negative log
    probabilities.
    
    Arguments:
        X -- Input image(s) from which predictions will be made (n x m)
        Y -- Target output(s) (actual labels) (10 x m)
        W -- Weight matrix (n x 10)
    '''
    
    #p2.SimpleNetwork(W, X) gets the prediction for X using the weights W

    return np.dot(X, (p2.SimpleNetwork(W, X) - Y).transpose())

def negLogLossGrad_FiniteDifference(X, Y, W, h):
    '''
    negLogLossGrad_FiniteDifference returns the finite difference approximation
    for the gradient of the negative log loss with respect to weight w_(i, j),
    for all w_(i,j) in W.
   
    Arguments:
        X -- Input image(s) from which predictions will be made
        Y -- Target output(s) (actual labels)
        W -- Weight matrix
        h -- differential quantity
    '''
   
    W_old = W.copy()
    Gradient = np.zeros((len(W), len(W[0])))
    for row in range(len(W)):
        for col in range(len(W[0])):
            W = W_old.copy()
            W[row, col] += h
            Gradient[row, col] = (NLL(p2.SimpleNetwork(W, X), Y) - 
                                  NLL(p2.SimpleNetwork(W_old, X), Y)) / h
    return Gradient