from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
from hashlib import sha256

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray / 255
def make3Sets(RESOLUTION, train_size, val_size, test_size):
    '''This function creates three sets of processed images with RESOLUTIONxRESOLUTION
    resolution. The size of each set is also given as an argument.
    This function assumes there to be a 'actors.txt' that lists each actor and bounding
    box for their face, as well as an 'uncropped' folder with the images of actors indexed
    by their line position in actors.txt.
    '''
    if not os.path.exists("../Data/Faces/training set" + str(RESOLUTION)):
        os.makedirs("../Data/Faces/training set" + str(RESOLUTION))

    if not os.path.exists("../Data/Faces/validation set" + str(RESOLUTION)):
        os.makedirs("../Data/Faces/validation set" + str(RESOLUTION))

    if not os.path.exists("../Data/Faces/test set" + str(RESOLUTION)):
        os.makedirs("../Data/Faces/test set" + str(RESOLUTION))

    #Changing to grayscale and cropping
    act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    for a in act:
        name = a.split()[1].lower()
        i = 0
        total = 0
        for line in open("../Data/Faces/actors.txt"):
            if a in line:
                filename = name + str(i) + '.' + line.split()[4].split('.')[-1]

                if not os.path.isfile("../Data/Faces/uncropped/" + filename):
                    i += 1
                    continue
                try:
                    line_split = line.split('\t')
                    hash = line_split[-1]
                    Image = open("../Data/Faces/uncropped/" + filename, "rb").read()
                    if sha256(Image) != hash:
                        print("Hash doesn't match")
                        continue
                    face = imread("../Data/Faces/uncropped/" + filename)
                    Coords = line_split[4].split(',')
                    cropped = face[int(Coords[1]):int(Coords[3]), int(Coords[0]):int(Coords[2])]
                    resized = imresize(cropped, (RESOLUTION, RESOLUTION))

                    if(resized.shape == (RESOLUTION*1L, RESOLUTION*1L, 3L) or resized.shape == (RESOLUTION*1L, RESOLUTION*1L, 4L)):
                        processed = rgb2gray(resized)
                    else:
                        processed = resized

                    savename = name + str(total) + '.' + line.split()[4].split('.')[-1]


                    if total < train_size:
                        imsave("../Data/Faces/training set" + str(RESOLUTION) + "/" + savename, processed, cmap = 'gray')
                    elif total < (train_size + test_size):
                        imsave("../Data/Faces/validation set" + str(RESOLUTION) + "/" + savename, processed, cmap='gray')
                    elif total < (train_size + test_size + val_size):
                        imsave("../Data/Faces/test set" + str(RESOLUTION) + "/" + savename, processed, cmap='gray')
                    else:
                        break
                    total += 1
                except IOError:
                    print 'file not valid'
                i += 1

RESOLUTION = 32
train_size = 70
val_size = 20
test_size = 20
make3Sets(RESOLUTION, train_size, val_size, test_size)