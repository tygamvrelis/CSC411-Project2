## part5.py
# In this file, a function is defined that implements gradient descent with
# momentum

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
import re

import part2 as p2
import part3 as p3

def part5_gradient_descent(X, Y, init_W, alpha, eps, max_iter, init_v, momentum):
    '''
    part5_gradient_descent finds a local minimum of the hyperplane defined by
    the hypothesis dot(W.T, X). The algorithm terminates when successive
    values of W differ by less than eps (convergence), or when the number of
    iterations exceeds max_iter. This function uses momentum
    
    Arguments:
        X -- input data for X (the data to be used to make predictions)
        Y -- input data for X (the actual/target data)
        init_W -- the initial guess for the local minimum (starting point)
        alpha -- the learning rate; proportional to the step size
        eps -- used to determine when the algorithm has converged on a 
               solution
        max_iter -- the maximum number of times the algorithm will loop before
                    terminating
        init_v -- the initial momentum value
        momentum -- the momentum parameter in range (0, 1)
    '''
    
    iter = 0
    previous_W = 0
    current_W = init_W.copy()
    previous_v = 0
    current_v = init_v # Initial momentum...
    firstPass = True
    history = list()
    Whistory = list()

    m = Y.shape[1]
    
    # Do-while...
    while(firstPass or
            (np.linalg.norm(current_W  - previous_W) > eps and 
            iter < max_iter)):
        firstPass = False
        
        previous_W = current_W.copy() # Update the previous W value
        previous_v = current_v # Update momentum
        
        # Update W
        current_v = momentum * current_v + alpha * p3.negLogLossGrad(X, Y, current_W)
        current_W = current_W - current_v
        
        if(iter % (max_iter // 100) == 0):
            # Print updates every so often and save cost into history list
            cost = p3.NLL(p2.SimpleNetwork(current_W, X), Y)
            history.append((iter, cost))
            Whistory.append(current_W)
            print("Iter: ", iter, " | Cost: ", cost)
            
        iter += 1
    
    return(Whistory, history)
    
def part5_train(X, Y, indices, numImages, alpha, eps, max_iter, init_W, init_v, momentum):
    '''
    part5_train returns the parameter W fit to the data in trainingSet using
    numImages images per digit via gradient descent with momentum
    
    Arguments:
        X -- matrix of training examples whose columns correspond to images from
             which predictions are to be made
        Y -- matrix of labels whose i-th column corresponds to the actual/target
             output for the i-th column in X
        indices -- a list containing the starting indexes for the various digits
        numImages -- a numerical value specifying the number of images per digit
                     to be used for training
        alpha -- gradient descent "learning rate" parameter (proportional to 
                 step size)
        eps -- gradient descent parameter determining how tight the convergence
               criterion is
        init_W -- initial weight to be used as a guess (starting point)
        init_v -- the initial momentum value
        momentum -- the momentum parameter in range (0, 1)
    '''
    
    # Prepare data
    numDigits = len(indices) # 10
    x = np.zeros(shape = (X.shape[0], numImages * numDigits))
    y = np.zeros(shape = (Y.shape[0], numImages * numDigits))
    
    j = 0
    for j in range(numDigits):
        offset = [k[1] for k in indices if k[0] == j][0]
        for i in range(numImages):
            x[:, i + j * numImages] = X[:, i + offset] # data to predict upon (images)
            y[:, i + j * numImages] = Y[:, i + offset] # target/actual values (labels)
        j += 1
    
    # Run gradient descent
    return part5_gradient_descent(x, y, init_W, alpha, eps, max_iter, init_v, momentum)