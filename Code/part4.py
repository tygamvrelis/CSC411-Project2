## part4.py
# In this file, helper functions are defined for training the simple neural
# network in part 2 using vanilla gradient descent, and plotting learning
# curves.

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

def makeTrainingMatrices():
    '''
    makeTrainingMatrices returns 2 matrices:
        X -- the training matrix whose columns correspond to images
        Y -- the label matrix whose i-th column corresponds to the i-th target
             output
             
    Also returned is a list of tuples (digit, start index). This way, one can
    easily reference the images for each digit from within X and Y.
    '''
    
    M = loadmat("../Data/mnist_all.mat") # Load MNIST dataset
    
    numExamples = sum([len(M[k]) for k in M.keys() if "train" in k])
    
    # Pre-allocate space for matrices
    X = np.empty((28 * 28 + 1, numExamples)) # 28 * 28 + 1 = num_pixels + bias
    Y = np.empty((10, numExamples)) # 10 is the number of output classes
    
    indices = list()
    i = 0
    for k in M.keys():
        if("train" in k):
            # print(k) # Notice that digit 0,...,10 are not grabbed in order
            numImages = M[k].shape[0] # Number of images for the current digit
            digitNum = int(re.findall('\d', k)[0]) # The current digit
            indices.append((digitNum, i)) # Track the starting index for this
                                          # digit in the columns of X
            
            M[k] = np.true_divide(M[k], 255.0) # Normalize images
            M[k] = np.vstack((np.ones((1, numImages)), M[k].T)) # Stack 1s ontop
            
            X[:, i:i + numImages] = M[k].copy() # Put images in X matrix
            
            # Make the label for this set of images
            label = np.zeros((10, 1))
            label[digitNum] = 1
            Y[:, i:i + numImages] = label
            
            i += numImages
            
    return (X, Y, indices)
    
def makeTestMatrices():
    '''
    makeTestMatrices returns 2 matrices:
        X -- the test matrix whose columns correspond to images
        Y -- the label matrix whose i-th column corresponds to the i-th target
             output
             
    Also returned is a list of tuples (digit, start index). This way, one can
    easily reference the images for each digit from within X and Y.
    '''
    
    M = loadmat("../Data/mnist_all.mat") # Load MNIST dataset
    
    numExamples = sum([len(M[k]) for k in M.keys() if "test" in k])
    
    # Pre-allocate space for matrices
    X = np.empty((28 * 28 + 1, numExamples)) # 28 * 28 + 1 = num_pixels + bias
    Y = np.empty((10, numExamples)) # 10 is the number of output classes
    
    indices = list()
    i = 0
    for k in M.keys():
        if("test" in k):
            # print(k)
            numImages = M[k].shape[0] # Number of images for the current digit
            digitNum = int(re.findall('\d', k)[0]) # The current digit
            indices.append((digitNum, i)) # Track the starting index for this
                                          # digit in the columns of X
            
            M[k] = np.true_divide(M[k], 255.0) # Normalize images
            M[k] = np.vstack((np.ones((1, numImages)), M[k].T)) # Stack 1s ontop
            
            X[:, i:i + numImages] = M[k].copy() # Put images in X matrix
            
            # Make the label for this set of images
            label = np.zeros((10, 1))
            label[digitNum] = 1
            Y[:, i:i + numImages] = label
            
            i += numImages
            
    return (X, Y, indices)
    

def part4_gradient_descent(X, Y, init_W, alpha, eps, max_iter):
    '''
    part4_gradient_descent finds a local minimum of the hyperplane defined by
    the hypothesis dot(W.T, X). The algorithm terminates when successive
    values of theta differ by less than eps (convergence), or when the number of
    iterations exceeds max_iter.
    
    Arguments:
        X -- input data for X (the data to be used to make predictions)
        Y -- input data for X (the actual/target data)
        init_W -- the initial guess for the local minimum (starting point)
        alpha -- the learning rate; proportional to the step size
        eps -- used to determine when the algorithm has converged on a 
               solution
        max_iter -- the maximum number of times the algorithm will loop before
                    terminating
    '''
    
    iter = 0
    previous_W = 0
    current_W = init_W.copy()
    firstPass = True
    history = list()
    
    m = Y.shape[1]
    
    # Do-while...
    while(firstPass or
            (np.linalg.norm(current_W  - previous_W) > eps and 
            iter < max_iter)):
        firstPass = False
        
        previous_W = current_W.copy() # Update the previous theta value
        
        # Update theta
        current_W = current_W - alpha * p3.negLogLossGrad(X, Y, current_W)
        
        if(iter % (max_iter // 100) == 0):
            # Print updates every so often and save cost into history list
            cost = p3.NLL(p2.SimpleNetwork(current_W, X), Y)
            history.append((iter, cost))
            print("Iter: ", iter, " | Cost: ", cost)
            
        iter += 1
    
    return(current_W, history)
    
def part4_train(X, Y, indices, numImages, alpha, eps, max_iter, init_W):
    '''
    part4_train returns the parameter W fit to the data in trainingSet using
    numImages images per digit via gradient descent.
    
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
    return part4_gradient_descent(x, y, init_W, alpha, eps, max_iter)
    
    
def part4_classify(X, Y, W, size):
    '''
    part4_classify returns the average cost and percentage of correct
    classifications for the hypothesis np.dot(W.T, x), using the learned
    weights W and testing the images in the input set against the labels.
    
    Arguments:
        X -- the input image matrix from which predictions are to be made
        Y -- the label matrix which the predictions will be compared to
        W -- the learned parameters that will be used to make predictions
        size -- the size of the classification set to be tested
    '''
    
    correct = 0.0
    size = len(X[0])
    P = p2.SimpleNetwork(W, X)

    for i in range(size):
        highest = np.argmax(P[:, i])
        if Y[highest, i] == 1:
            correct += 1
    correct = correct/size
    cost = p3.NLL(P, Y)/size
    return (cost, correct)


def part4_plotLearningCurves(history):
    '''
    part4_plotLearningCurves plots the learning curves associated with training
    a neural network.
    
    Arguments:
        history -- a list of pairs of numbers (num_iterations, cost), where
                   cost is the average cost associated with training the neural
                   network using num_examples training examples.
    '''

    num_iter = [i[0] for i in history]
    cost = [i[1] for i in history]
    
    plt.plot(num_iter, cost)
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.title('Training Set Learning Curve')
    plt.show()
    
def part4_plotWeights(W, indices, imagePath):
    '''
    part4_plotWeights produces visualizations of the learned parameters in the
    weight matrix W.
    
    Arguments:
        W -- the weight matrix to be visualized
        indices -- a list containing the starting indexes for the various digits
        imagePath -- a string giving the location to which images should be saved
    '''
    
    nums = [k[0] for k in indices]
    for n in nums:
        plt.yscale('linear')
        plt.title("(Part 4) " + str(n))
        plt.imshow(W[1:,n].reshape((28,28)), interpolation = 'gaussian', cmap = plt.cm.coolwarm)
        plt.savefig(imagePath + "p4_" + str(n) + ".jpg")
        plt.gcf().clear()