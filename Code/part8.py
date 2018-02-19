import numpy as np
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
import os
from scipy.misc import imread

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

    image_size = RESOLUTION * RESOLUTION

    batch_xs = np.zeros((0, image_size * 3))
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

def train_set(trainX, trainY, valX, valY, model, steps, batch_size, loss_fn, optimizer):
    tloss_hist = []
    tperf_hist = []
    vloss_hist = []
    vperf_hist = []
    num_iter = []
    for t in range(steps):
        train_idx = np.random.permutation(range(trainX.shape[0]))[:batch_size]
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
    x = Variable(torch.from_numpy(X), requires_grad=False).type(dtype_float)
    y_classes = Variable(torch.from_numpy(np.argmax(Y, 1)), requires_grad=False).type(dtype_long)
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

def viewOutputLayer(X, model):
    '''
    viewOutputLayer returns a raw view of the output layer.

    Arguments:
        X -- the input to the network
        model -- the trained model
    '''

    dtype_float = torch.FloatTensor
    x = Variable(torch.from_numpy(X), requires_grad=False).type(dtype_float)
    y_pred = model(x)
    return y_pred

def viewHiddenLayer(X, y_ind, model, dim_x, dim_h):
    '''
    viewHiddenLayer returns the raw view of the hidden layer for a given input.

    Arguments:
        X -- the input to the network
        y_ind -- the output neuron whose input weights are to be examined
        model -- the trained model
        dim_x -- the dimension of the input
        dim_h -- the dimension of the hidden layer
    '''

    dtype_float = torch.FloatTensor
    x = Variable(torch.from_numpy(X), requires_grad=False).type(dtype_float)

    weights = model.__getitem__(2)._parameters['weight'][y_ind,:]
    bias = model.__getitem__(2)._parameters['bias'][y_ind]

    model2 = torch.nn.Sequential(
        torch.nn.Linear(dim_x, dim_h),
        torch.nn.Tanhshrink(),
    )
    model2.classifier = torch.nn.Sequential(*list(model.children())[:-1])

    hidden = model2.classifier(x)
    for i in range(weights.shape[1]):
        hidden[0, i] = hidden[0, i] * weights[0, i]

    return hidden

def viewWeights(hiddenUnitIndex, model, res, imagePath, actorName, showch = 0):
    '''
    viewWeights provides a visualization of the weights going into the specified
    hidden neuron. The positive and negative weights are displayed separately since
    the plot function cannot interpret negative pixel values for images.

    Arguments:
        hiddenUnitIndex -- the hidden unit whose input weights are to be examined
        model -- the trained model
        res -- the resolution of the images
        imagePath -- a string indicating where to save plots
        actorName -- the name of the actor (to be used while saving images)
        showch -- if 1, then the colors channels for the weight visualization are
                  plotted separately; otherwise, the weights are plotted in color
    '''

    W = model.__getitem__(0)._parameters['weight'][hiddenUnitIndex, :].cpu().data.numpy()

    Wpos = W.copy() # Positive weights
    for i in range(Wpos.shape[0]):
        Wpos[i] = Wpos[i] if Wpos[i] > 0 else 0

    Wneg = Wpos - W # Negative weights

    Wpos /= np.max(Wpos)
    Wneg /= np.max(Wneg)
    Wpos = np.rot90(Wpos.reshape((res, res, 3), order = "F"), 1, (1, 0))
    Wneg = np.rot90(Wneg.reshape((res, res, 3), order = "F"), 1, (1, 0))

    if(showch):
        for i in range(3):
            plt.figure(i)
            plt.imshow(Wpos[:, :, i],  cmap = plt.cm.coolwarm)
            plt.title('Useful weights for ' + actorName + " (positive values)" + "(Ch: " + str(i) + ")")
            plt.show()
            #plt.savefig(imagePath + "part8" + actorName + ".jpg")

        for i in range(3):
            plt.figure(3+i)
            plt.imshow(Wneg[:, :, i],  cmap = plt.cm.coolwarm)
            plt.title('Useful weights for ' + actorName + " (negative values)" + "(Ch: " + str(i) + ")")
            plt.show()
            #plt.savefig(imagePath + "part8" + actorName + ".jpg")
    else:
        plt.figure(1)
        plt.imshow(Wpos, interpolation = 'gaussian')
        plt.title('Useful weights for ' + actorName + " (positive values)")
        plt.show()
        # plt.savefig(imagePath + "part8" + actorName + ".jpg")

        plt.figure(2)
        plt.imshow(Wneg,  interpolation = 'gaussian')
        plt.title('Useful weights for ' + actorName + " (negative values)")
        plt.show()
        # plt.savefig(imagePath + "part8" + actorName + ".jpg")

    #plt.gcf().clear()