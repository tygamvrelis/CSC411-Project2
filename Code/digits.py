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
import part2 as p2
    
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
    plt.imshow(X.reshape((28,28)))
    plt.show()
    y = p2.SimpleNetwork(W, X)
    print("y: ", y) # Should be a 10x1 vector with random values
    print("sum(y): ", sum(y)) # Should be 1

##  Part 3: Negative log loss of gradient
import part3 as p3
   
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
    G1 = p3.negLogLossGrad(X, Y, W)
    h = 1.0
    for i in range(10):
        G2 = p3.negLogLossGrad_FiniteDifference(X, Y, W, h)
        print "Total error: ", sum(abs(G1 - G2)), "h: ", h
        h /= 10

##  Part 4: Training using vanilla gradient descent
import part4 as p4

# Train and classify while increasing the number of images per digit
# store the results in a list so that we can plot them later
    p4_history = list()
    
    # We can only increase the training set to be as large as the smallest
    # training set
    size = min(len(M[k]) for k in M.keys() if "train" in k)
    for i in range(size):
        # print("PART 5: i = ", i)
        # (theta, history) = p5.part5_train(actList, i + 1, 1.60E-5, 5E-7, 600000)
        # (cost_actList, corr_actList) = p5.part5_classify(actList, theta, i + 1) # Test on training set
        # if(i < 60):
        #     (cost_val_actList, corr_val_actList) = p5.part5_classify(val_actList, theta, i + 1) # Test on validation set
        #     (cost_val_not_actList, corr_val_not_actList) = p5.part5_classify(val_not_actList, theta, i + 1) # Test on validation set for 6 actors not in act
        # else:
        #     (cost_val_actList, corr_val_actList) = p5.part5_classify(val_actList, theta, 60) # Test on validation set
        #     (cost_val_not_actList, corr_val_not_actList) = p5.part5_classify(val_not_actList, theta, 60) # Test on validation set for 6 actors not in act
        # 
        # p5_history.append((cost_actList, corr_actList, cost_val_actList, corr_val_actList,
        #                    cost_val_not_actList, corr_val_not_actList))

## Others
def tanh_layer(y, W, b):    
    '''
    Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases
    '''
    
    return tanh(dot(W.T, y)+b)

def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0) # tanh layer output
    L1 = dot(W1.T, L0) + b1 # inner product layer output
    output = softmax(L1)
    return L0, L1, output

def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''
    Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network
    '''
    
    dCdL1 =  y - y_
    dCdW1 =  dot(L0, dCdL1.T ) 