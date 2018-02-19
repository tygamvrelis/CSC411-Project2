import numpy as np
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
from scipy.misc import imread

#matplotlib inline
M = loadmat("../Data/mnist_all.mat")

def processActorString(filename):
    '''This function takes the name of an image file and returns its actor and pic number.
     The file name is assumed to have the naming convention: 'actor#.ext'.
    '''
    actor = filename.replace('.jpg', '')
    actor = actor.replace('.jpeg', '')
    actor = actor.replace('.JPG', '')
    actor = actor.replace('.png', '')
    temp_actor = actor

    for char in '0123456789.':
        actor = actor.replace(char, '')
    pic_number = int(temp_actor.replace(actor, ''))
    return (actor, pic_number)

def get_set(file, RESOLUTION, act):
    '''This function creates input and label matrices of size [# of samples]x[RESOLUTION^2*3]
    from all of the images in a given file path. The images are all assumed to have the naming
    convention: 'actor#.ext' with the actors being one of those in the act array.
    The output label matrix is one hot encoded.
    '''
    image_size = RESOLUTION*RESOLUTION

    batch_xs = np.zeros((0, image_size* 3))
    batch_y_s = np.zeros((0, 6))


    for filename in os.listdir(file):
        face = imread(file + '/' + filename)
        (actor, throw_away) = processActorString(filename)
        input = np.concatenate((np.array(face[:, :, 0]).reshape(image_size), np.array(face[:, :, 1]).reshape(image_size), np.array(face[:, :, 2]).reshape(image_size)))
        batch_xs = np.vstack((batch_xs, (input / 255.)))

        one_hot = np.zeros(6)
        one_hot[act.index(actor)] = 1

        batch_y_s = np.vstack((batch_y_s, one_hot))
    return batch_xs, batch_y_s

def train_set(trainX, trainY, valX, valY, model, steps, loss_fn, optimizer):
    tloss_hist = []
    tperf_hist = []
    vloss_hist = []
    vperf_hist = []
    num_iter = []
    for t in range(steps):
        train_idx = np.random.permutation(range(trainX.shape[0]))[:100]
        (loss, perf) = classify(trainX[train_idx], trainY[train_idx], model, loss_fn)

        model.zero_grad()  # Zero out the previous gradient computation
        loss.backward()  # Compute the gradient
        optimizer.step()  # Use the gradient information to
        # make a step

        if (t % (steps // 100) == 0):
            # Print updates every so often and save cost into history list
            (loss, perf) = classify(trainX, trainY, model, loss_fn)
            tloss_hist.append(loss.cpu().data)
            tperf_hist.append(perf)

            (loss, perf) = classify(valX, valY, model, loss_fn)
            vloss_hist.append(loss.cpu().data)
            vperf_hist.append(perf)
            num_iter.append(t)

    return (tloss_hist,tperf_hist, vloss_hist, vperf_hist, num_iter, model)


def classify(X, Y, model, loss_fn):
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor
    x = Variable(torch.from_numpy(X), requires_grad=False).type(dtype_float).cuda()
    y_classes = Variable(torch.from_numpy(np.argmax(Y, 1)), requires_grad=False).type(dtype_long).cuda()
    y_pred = model(x)
    loss = loss_fn(y_pred, y_classes)
    perf = np.mean(np.argmax(y_pred.cpu().data.numpy(), 1) == np.argmax(Y, 1))

    return (loss, perf)

def draw_curves(tloss_hist,tperf_hist, vloss_hist, vperf_hist, num_iter):
    plt.figure(1)
    plt.plot(num_iter, tloss_hist)
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.title('Training Set Cost Learning Curve')

    plt.figure(2)
    plt.plot(num_iter, tperf_hist)
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.title('Training Set Accuracy Learning Curve')

    plt.figure(3)
    plt.plot(num_iter, vloss_hist)
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.title('Validation Set Cost Learning Curve')

    plt.figure(4)
    plt.plot(num_iter, vperf_hist)
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.title('Validation Set Accuracy Learning Curve')
    plt.show()

    plt.gcf().clear()