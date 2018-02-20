## Faces.py
# This file contains the code to run parts 8 through 10 for CSC411 project 2.
# This involves running machine learning algorithms on faces from the FaceScrub dataset.

from pylab import *
import numpy as np
import torch
from torch.autograd import Variable

## Part 8: Using PyTorch to train a single-hidden-layer fully-connected NN to classify faces
import part8 as p8
import image_processing as improc
np.random.seed(0)

torch.manual_seed(0)
# Note: The act array determines the index of each actor in the one-hot encoding.
act = ['bracco', 'gilpin', 'harmon', 'baldwin', 'hader', 'carell']

dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor
RESOLUTION = 28
train_size = 70
val_size = 20
test_size = 20

#improc.make3Sets(RESOLUTION, train_size, val_size, test_size)

(trainX, trainY) = p8.get_set("../Data/Faces/training set" + str(RESOLUTION), RESOLUTION, act)
(valX, valY) = p8.get_set("../Data/Faces/validation set" + str(RESOLUTION), RESOLUTION, act)
(testX, testY) = p8.get_set("../Data/Faces/test set" + str(RESOLUTION), RESOLUTION, act)


dim_x = RESOLUTION ** 2 * 3
dim_h = 70
dim_out = 6
model = torch.nn.Sequential(
    torch.nn.Linear(dim_x, dim_h),
    torch.nn.Tanh(),
    torch.nn.Linear(dim_h, dim_out),
)
if torch.cuda.is_available():
    model = model.cuda()
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
steps = 500
batch_size = 60

(tloss_hist,tperf_hist, vloss_hist, vperf_hist, num_iter, model) = p8.train_set(trainX, trainY, valX, valY, model, steps, batch_size, loss_fn, optimizer)


(loss, perf) = p8.classify(valX, valY, model, loss_fn)
print "Classification Val performance: ", perf
(loss, perf) = p8.classify(testX, testY, model, loss_fn)
print "Classification Test performance: ", perf
#82.5% on both sets

p8.draw_curves(tloss_hist,tperf_hist, vloss_hist, vperf_hist, num_iter)


# ## Part 9: Visualize the weights of the hidden units that are useful for classifying each actor
# # Define "useful" to mean the weights from the hidden layer to the output layer with the
# # largest maximum value, once multiplied by its activation
# imagePath = "../Report/images/"
#
# # ind selects an image in the validation set. IF this image is classified correctly,
# # it will be fed into the network. The two neurons in the hidden layer that have the
# # most positive and most negative outputs given this image are then taken to be the
# # most "useful" neurons in classifying photos of this actor. The weights connecting
# # the input image to these two neurons are then displayed in color.
# ind = 100
# x = valX[ind:ind+1,:]
#
# # If this image is correctly classified, we can proceed
# output = p8.viewOutputLayer(x, model)
# if np.argmax(output.cpu().data.numpy(), 1) == np.argmax(valY[ind:ind+1,:], 1):
#     y_ind = np.argmax(valY[ind:ind+1,:], 1)
#     hidden = p8.viewHiddenLayer(x, y_ind, model, dim_x, dim_h)
#
#     # Select the "useful" hidden units
#     hidden = hidden.cpu().data.numpy()
#     maxNeuron = np.argmax(hidden)
#     minNeuron = np.argmin(hidden)
#
#     # Visualize
#     p8.viewWeights(maxNeuron, model, RESOLUTION, imagePath, act[int(y_ind)])
