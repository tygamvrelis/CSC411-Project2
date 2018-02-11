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
import part3 as p3
import part2 as p2

def plot_contour(W, X, Y, i1, j1, i2, j2):
    '''
    Contour map takes a weight matrix, W, and the coordinates for 2 weights.
    It plots a contour map of the NLL function with respect to X, W and Y by
    varying the weights w1 and w2.

    W:  The weight matrix of size nx10
    X:  A matrix of data of size nxm. Should use training set.
    Y:  A matrix of labels for the X matrix, of size 10xm
    i1, j1: Coordinates for w1 in the weight matrix s.t. w1 = W[i1, j1]
    i2, j2: Coordinates for w2 in the weight matrix s.t. w2 = W[i2, j2]
    '''
    delta = 0.1

    w1 = np.arange(-3, 0, delta)
    w2 = np.arange (-1, 3, delta)
    Wchanged = W.copy()

    W1, W2 = np.meshgrid(w1, w2)
    w1_Width = int(w1.shape[0])
    w2_Width = int(w2.shape[0])
    Z = np.empty((w2_Width, w1_Width))
    for i in range(w1_Width):
        for j in range(w2_Width):
            Wchanged[i1, j1] = w1[i]
            Wchanged[i2, j2] = w2[j]
            Z[j , i] = p3.NLL(p2.SimpleNetwork(Wchanged, X), Y)/int(X.shape[1])


    plt.figure(5)
    plt.contour(W1, W2, Z)
    #plt.plot([-1, 0, 1], [1, 0, -1])
    plt.title('Contour Map')
    plt.show()
    
def k_steps_gradient_descent(X, Y, init_W, alpha, k, momentum = 0, i1, j1, i2, j2):
    '''
    k_steps_gradient_descent finds a local minimum of the hyperplane defined by
    the hypothesis dot(W.T, X). The algorithm terminates when k iterations have
    been performed.
    
    Arguments:
        X -- input data for X (the data to be used to make predictions)
        Y -- input data for X (the actual/target data)
        init_W -- the initial guess for the local minimum (starting point)
        alpha -- the learning rate; proportional to the step size
        k -- the number of iterations to be performed
        momentum -- the momentum parameter in range (0, 1)
        i1, j1: Coordinates for w1 in the weight matrix s.t. w1 = W[i1, j1]
        i2, j2: Coordinates for w2 in the weight matrix s.t. w2 = W[i2, j2]
    '''
    Whistory = list()
    iter = 0
    previous_W = 0
    current_W = init_W.copy()
    previous_v = 0
    current_v = init_v # Initial momentum...
    
    # Do-while...
    while(firstPass or iter < k):
        firstPass = False
        
        previous_W = current_W.copy() # Update the previous W value
        
        # Update W and v
        current_v = momentum * current_v + alpha * p3.negLogLossGrad(X, Y, current_W)
        current_W[i1, j1] = current_W[i1, j1] - current_v[i1, j1]
        current_W[i2, j2] = current_W[i2, j2] - current_v[i2, j2]
        

        Whistory.append(current_W)
            
        iter += 1
    
    return(Whistory)