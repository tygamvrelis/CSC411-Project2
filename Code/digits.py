## digits.py
# This file contains the code to run parts 1 through 7 for CSC411 project 2.
# This involves running machine learning algorithms on the MNIST dataset

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


## Part 1: Describing the dataset
# Load the MNIST digit data
M = loadmat("../Data/mnist_all.mat")

def part1SaveImages():
    '''
    part1SaveImages saves a sample of images from the MNIST dataset into the
    images folder for the report.
    '''
    
    for k in range(10):
        for i in range(10): #Display the 150-th "5" digit from the training set
            imshow(M["train" + str(k)][i].reshape((28,28)), cmap=cm.gray)
            imsave("../Report/images/number" + str(k) + "_"+ str(i) + ".jpg", 
                   M["train" + str(k)][i].reshape((28,28)),
                   cmap=cm.gray)

train = zeros(10)
test = zeros(10)
for i in range(10):
    train[i] = len(M["train" + str(i)])
    test[i] = len(M["test" + str(i)])

# Normalize images (map to [0, 1])
for k in M.keys():
    if("train" in k or "test" in k):
        M[k] = np.true_divide(M[k], 255.0)
        
## Part 2: Computing a simple network
def softmax(y):
    '''
    Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases
    '''
    
    return exp(y)/tile(sum(exp(y),0), (len(y),1))
    
def SimpleNetwork(theta, X):
    '''
    SimpleNetwork returns the vectorized multiplication of the (n x 10) 
    parameter matrix W with the data X.
    
    Arguments:
        W -- (n x 10) matrix of parameters (weights and biases)
        x -- (n x m) matrix whose i-th column corresponds to the i-th training 
             image
    '''
    
    return softmax(np.dot(theta.T, X))
    
def part2():
    '''
    testPart2 uses a randomly-initialized weight matrix to make a prediction
    about an image from the MNIST dataset (which happens to be 5). The image
    is plotted and the output of the SimpleNetwork computation from part 2 is
    printed.
    '''
    
    np.random.seed(3)
    W = np.random.rand(28*28,10) # Randomly initialize some weight matrix
    X = M["train5"][148:149].T # Image of "5"
    plt.imshow(x.reshape((28,28)))
    plt.show()
    y = SimpleNetwork(W, X)
    print("y: ", y) # Should be a 10x1 vector with random values
    print("sum(y): ", sum(y)) # Should be 1

##  Part 3: Negative log loss of gradient
def negLogLossGrad(X, Y, W):
    '''
    negLogLossGrad returns the gradient of the cost function of negative log
    probabilities.
    
    Arguments:
        X -- Input image(s) from which predictions will be made (n x m)
        Y -- Target output(s) (actual labels) (10 x m)
        W -- Weight matrix (n x 10)
    '''
    
    P = SimpleNetwork(W, X) # Get the prediction for X using the weights W

    return np.dot(X, (P - Y).transpose())

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
            Gradient[row, col] = (NLL(SimpleNetwork(W, X), Y) - NLL(SimpleNetwork(W_old, X), Y)) / h
    return Gradient
   
def part3():
    '''
    part3Test computes the vectorized gradient as well as a finite difference
    approximation for the gradient of the negative log loss for 10 different
    values of h (the differential quantity). The differences between all the 
    gradient matrix entries for the two methods are summed and printed for each
    h-value.
    '''
    
    X = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3 ,4],[1, 2, 3, 4]]
    X = np.array(X)
    W = [[1, 0, 1.2, 3], [1, 2, 0.2, 1.2], [1, 8, 1, 4], [1, -2, 3, 1], [1, 0, 3, 1]]
    W = np.array(W)
    Y = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    Y = np.array(Y)
    G1 = negLogLossGrad(X, Y, W)
    h = 1.0
    for i in range(10):
        G2 = negLogLossGrad_FiniteDifference(X, Y, W, h)
        print "Total error: ", sum(abs(G1 - G2)), "h: ", h
        h /= 10

##  Part 4: Training using vanilla gradient descent


## Others
def tanh_layer(y, W, b):    
    '''
    Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases
    '''
    
    return tanh(dot(W.T, y)+b)

def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = dot(W1.T, L0) + b1
    output = softmax(L1)
    return L0, L1, output
    
def NLL(P, Y):
    return -sum(Y*log(P)) 

def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''
    Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network
    '''
    
    dCdL1 =  y - y_
    dCdW1 =  dot(L0, dCdL1.T ) 