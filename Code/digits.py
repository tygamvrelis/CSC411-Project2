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

def testPart2():
    '''
    testPart2 uses a randomly-initialized weight matrix to make a prediction
    about an image from the MNIST dataset (which happens to be 5). The image
    is plotted and the output of the SimpleNetwork computation from part 2 is
    printed.
    '''

    np.random.seed(3)
    W = np.random.rand(28*28,10) # Randomly initialize some weight matrix
    X = M["train5"][148:149].T # Image of "5"
    #plt.imshow(X.reshape((28,28)))
    #plt.show()
    y = p2.SimpleNetwork(W, X)
    print("y: ", y) # Should be a 10x1 vector with random values
    print("sum(y): ", sum(y)) # Should be 1
testPart2()


##  Part 3: Negative log loss of gradient
import part3 as p3

def testPart3():
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
testPart3()

##  Part 4 and 5: Training using vanilla vs. momentum gradient descent
import part4 as p4
import part5 as p5

# Make training and test matrices
(X, Y, indicesTrain) = p4.makeTrainingMatrices()
(XTest, YTest, indicesTest) = p4.makeTestMatrices()
train_size = 1000 # Number of images per digit to use for training
val_size = 100
(XTrain, YTrain, XVal, YVal) = p4.part4_split_sets(X, Y, train_size, val_size, indicesTrain)

# Gradient descent
np.random.seed(3)
init_W = np.random.rand(28*28 + 1,10) # Randomly initialize weight matrix
alpha = 1e-5
eps = 1e-6
max_iter = 500


(Whistory, history) = p4.part4_gradient_descent(XTrain, YTrain, init_W, alpha, eps, max_iter)


# Plot learning curves
imagePath = "../Report/images/"
p4.part4_plotLearningCurves(XTrain, YTrain, XVal, YVal, Whistory, history, imagePath, "p4_")


#Plot weight visualizations
imagePath = "../Report/images/"
p4.part4_plotWeights(Whistory[99], indicesTrain, imagePath, "p4_")



#Now do it for momentum
init_v = 0.1 # Initial momentum value
momentum = 0.9

(Whistory, history) = p5.part5_gradient_descent(XTrain, YTrain, init_W, alpha, eps, max_iter, init_v, momentum)
# np.savetxt("Weights", Whistory[99])
# W = np.loadtxt("Weights")
W = Whistory[99]


#Plot learning curves
imagePath = "../Report/images/"
p4.part4_plotLearningCurves(XTrain, YTrain, XVal, YVal, Whistory, history, imagePath, "p5_")

#Plot weight visualizations
imagePath = "../Report/images/"
p4.part4_plotWeights(W, indicesTrain, imagePath, "p5_")


##  Part 6: Contour Plot
import part6 as p6
import part4 as p4

# Make training and test matrices
(X, Y, indicesTrain) = p4.makeTrainingMatrices()
(XTest, YTest, indicesTest) = p4.makeTestMatrices()
train_size = 1000 # Number of images per digit to use for training
val_size = 100
(XTrain, YTrain, XVal, YVal) = p4.part4_split_sets(X, Y, train_size, val_size, indicesTrain)

# Gradient descent
np.random.seed(3)
init_W = np.random.rand(28*28 + 1,10) # Randomly initialize weight matrix
alpha = 1e-5
eps = 1e-6
(i1, j1) = (200, 2)
(i2, j2) = (201, 2)


k = 20
alpha = 0.5
momentum = 0.9
(w1_van, w2_van) = p6.k_steps_gradient_descent(XTrain, YTrain, init_W, alpha, k, i1, j1, i2, j2)
(w1_mom, w2_mom) = p6.k_steps_gradient_descent(XTrain, YTrain, init_W, alpha, k, i1, j1, i2, j2, momentum)
p6.plot_contour(init_W, XTrain, YTrain, i1, j1, i2, j2, w1_van, w2_van, w1_mom, w2_mom)


alpha = 0.5
momentum = 0.3
(w1_van, w2_van) = p6.k_steps_gradient_descent(XTrain, YTrain, init_W, alpha, k, i1, j1, i2, j2)
(w1_mom, w2_mom) = p6.k_steps_gradient_descent(XTrain, YTrain, init_W, alpha, k, i1, j1, i2, j2, momentum)
p6.plot_contour(init_W, XTrain, YTrain, i1, j1, i2, j2, w1_van, w2_van, w1_mom, w2_mom)


alpha = 0.55
momentum = 0.3
(w1_van, w2_van) = p6.k_steps_gradient_descent(XTrain, YTrain, init_W, alpha, k, i1, j1, i2, j2)
(w1_mom, w2_mom) = p6.k_steps_gradient_descent(XTrain, YTrain, init_W, alpha, k, i1, j1, i2, j2, momentum)
p6.plot_contour(init_W, XTrain, YTrain, i1, j1, i2, j2, w1_van, w2_van, w1_mom, w2_mom)