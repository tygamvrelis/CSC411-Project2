
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
#Load the MNIST digit data
M = loadmat("../Data/mnist_all.mat")

# for k in range(10):
#     for i in range(10): #Display the 150-th "5" digit from the training set
#         imshow(M["train" + str(k)][i].reshape((28,28)), cmap=cm.gray)
#         imsave("../Report/images/number" + str(k) + "_"+ str(i) + ".jpg", M["train" + str(k)][i].reshape((28,28)), cmap=cm.gray)

train = zeros(10)
test = zeros(10)
for i in range(10):
    train[i] = len(M["train" + str(i)])
    test[i] = len(M["test" + str(i)])
print train

print test
## Part 2: Computing a simple network
# Normalize images (map to [0, 1])
for k in M.keys():
    if("train" in k or "test" in k):
        M[k] = np.true_divide(M[k], 255.0)

def softmax(y):
    '''
    Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases
    '''
    
    return exp(y)/tile(sum(exp(y),0), (len(y),1))
######
def Part2(theta, X):
    '''
    Part2 returns the vectorized multiplication of the (n x 10) parameter matrix 
    theta with the data X.
    
    Arguments:
        theta -- (n x 10) matrix of parameters (weights and biases)
        x -- (n x 1) matrix whose rows correspond to pixels in images
    '''
    
    return softmax(np.dot(theta.T, X))
    
# Part 2 test code
np.random.seed(3)
theta = np.random.rand(28*28,10) # Randomly initialize some theta matrix
x = M["train5"][148:149].T # Image of "5"
plt.imshow(x.reshape((28,28)))
plt.show()
y = Part2(theta, x)
print("y: ", y) # Should be a 10x1 vector with random values
print("sum(y): ", sum(y)) # Should be 1


####PART 3#############
def negLogLossGrad(X, Y, W):
    P = Part2(W, X)

    return np.dot(X, (P - Y).transpose())
P = [[1, 4],[2, 5],[3, 6]]
Y = [[1, 0], [0, 0], [0, 1]]
P = np.array(P)
Y = np.array(Y)
#######################3
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
    
def NLL(y, y_):
    return -sum(y_*log(y)) 

def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''
    Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network
    '''
    
    dCdL1 =  y - y_
    dCdW1 =  dot(L0, dCdL1.T ) 
    

#Load sample weights for the multilayer neural network
snapshot = cPickle.load(open("../Data/snapshot50.pkl"))
W0 = snapshot["W0"]
b0 = snapshot["b0"].reshape((300,1))
W1 = snapshot["W1"]
b1 = snapshot["b1"].reshape((10,1))

#Load one example from the training set, and run it through the
#neural network
x = M["train5"][148:149].T    
L0, L1, output = forward(x, W0, b0, W1, b1)
#get the index at which the output is the largest
y = argmax(output)

################################################################################
#Code for displaying a feature from the weight matrix mW
#fig = figure(1)
#ax = fig.gca()    
#heatmap = ax.imshow(mW[:,50].reshape((28,28)), cmap = cm.coolwarm)    
#fig.colorbar(heatmap, shrink = 0.5, aspect=5)
#show()
################################################################################
