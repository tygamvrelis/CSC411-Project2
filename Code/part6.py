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
    # print Z
    #levels = np.arange(np.min(Z), 2*np.min(Z), np.min(Z)/2)

    plt.figure(5)
    plt.contour(W1, W2, Z)
    plt.title('Contour Map')
    plt.show()