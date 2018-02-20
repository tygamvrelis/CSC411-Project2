from pylab import *
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cbook as cbook
# import random
# import time
from scipy.misc import imread
from scipy.misc.pilutil import imresize
from scipy.misc import imsave
import os
from hashlib import sha256
import part8 as p8

#Hashing seems to take out some valid images so I hard code them back in here.
valid_gilpin = [2, 3, 15, 19, 24, 41, 43, 58, 62, 68, 70, 76, 84, 109, 117, 122, 135, 137, 139, 140, 141, 143, 146, 153, 154, 159, 164, 167, 170 , 175, 194]
valid_harmon = [4, 16, 18, 30, 37, 38, 52, 56, 58, 78, 82, 87, 93, 99, 101, 116, 155, 166, 168, 177, 181]

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

    badHashCount = 0
    g = 0
    invalidFiles = 0
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
                    invalidFiles += 1
                    continue
                try:
                    line_split = line.split('\t')
                    expectedHash = line_split[-1].rstrip()
                    img = open("../Data/Faces/uncropped/" + filename, "rb").read()
                    computedHash = sha256(img).hexdigest()
                    if computedHash != expectedHash:
                        remove = True
                        (actor, number) = p8.processActorString(filename)
                        if actor == "gilpin" and number in valid_gilpin:
                             remove = False
                        if actor == "harmon" and number in valid_harmon:
                            remove = False
                        if remove:
                            print("Hash doesn't match! File: " + filename)
                            badHashCount += 1
                            i += 1
                            continue
                    face = imread("../Data/Faces/uncropped/" + filename)
                    try:
                        face.shape[2]
                    except:
                        print("grayscale")
                        g += 1
                        i += 1
                        continue
                    Coords = line_split[4].split(',')
                    cropped = face[int(Coords[1]):int(Coords[3]), int(Coords[0]):int(Coords[2])]
                    resized = imresize(cropped, (RESOLUTION, RESOLUTION))

                    # if(resized.shape == (RESOLUTION*1L, RESOLUTION*1L, 3L) or resized.shape == (RESOLUTION*1L, RESOLUTION*1L, 4L)):
                    #     processed = rgb2gray(resized)
                    # else:
                    #     processed = resized

                    savename = name + str(total) + '.' + line.split()[4].split('.')[-1]


                    if total < train_size:
                        imsave("../Data/Faces/training set" + str(RESOLUTION) + "/" + savename, resized)
                    elif total < (train_size + test_size):
                        imsave("../Data/Faces/validation set" + str(RESOLUTION) + "/" + savename, resized)
                    elif total < (train_size + test_size + val_size):
                        imsave("../Data/Faces/test set" + str(RESOLUTION) + "/" + savename, resized)
                    else:
                        break
                    total += 1
                except IOError:
                    print 'file not valid'
                i += 1


    print "Number of mismatched hashes: ", badHashCount
    minNumTraining = ("", sys.maxsize)
    minNumValidation = ("", sys.maxsize)
    minNumTest = ("", sys.maxsize)
    for actor in act:
        a = actor.split()[1].lower()
        numTraining = len([name for name in os.listdir("../Data/Faces/training set" + str(RESOLUTION)) if a in name])
        numValidation = len([name for name in os.listdir("../Data/Faces/validation set" + str(RESOLUTION)) if a in name])
        numTest = len([name for name in os.listdir("../Data/Faces/test set" + str(RESOLUTION)) if a in name])
        print("Actor: " + a +
              "| #training: " + str(numTraining) +
              "| #validation: " + str(numValidation) +
              "| #Test: " + str(numTest)
              )
        if(numTraining < minNumTraining[1]):
            minNumTraining = (a, numTraining)
        if (numValidation < minNumValidation[1]):
            minNumValidation = (a, numValidation)
        if (numTest < minNumTest[1]):
            minNumTest = (a, numTest)
    print("Min #training: " + str(minNumTraining))
    print("Min #validation: " + str(minNumValidation))
    print("Min #test: " + str(minNumTest))
    print("Grayscale images: " + str(g))
    print("Unopenable Files " + str(invalidFiles))
