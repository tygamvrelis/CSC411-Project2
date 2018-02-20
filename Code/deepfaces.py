## Part 10: Using activations from the Conv4 layer in AlexNet to train a face classifier
import myalexnet as myAN
import part8 as p8
import part10 as p10
import image_processing as improc
from pylab import *
import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)
modelAN = myAN.MyAlexNet()

if torch.cuda.is_available():
    modelAN = modelAN.cuda()
modelAN.eval()

act = ['bracco', 'gilpin', 'harmon', 'baldwin', 'hader', 'carell']
dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor
RESOLUTION = 227
train_size = 70
val_size = 20
test_size = 20


#improc.make3Sets(RESOLUTION, train_size, val_size, test_size)

# Get images in 3D form (not flattened, since AlexNet requires the 3D form to convolve with filters)
(trainX, trainY) = p8.get_set("../Data/Faces/training set" + str(RESOLUTION), RESOLUTION, act, noflatten = 1)
(valX, valY) = p8.get_set("../Data/Faces/validation set" + str(RESOLUTION), RESOLUTION, act, noflatten = 1)
(testX, testY) = p8.get_set("../Data/Faces/test set" + str(RESOLUTION), RESOLUTION, act, noflatten = 1)


trainX_feat = p10.get_feature_set(trainX, train_size*6 ,RESOLUTION, modelAN)
valX_feat = p10.get_feature_set(valX, val_size*6 ,RESOLUTION, modelAN)
testX_feat = p10.get_feature_set(testX, test_size*6 ,RESOLUTION, modelAN)
dim_x = trainX_feat.shape[1]
dim_h = 50
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
steps = 200
batch_size = 50

(tloss_hist,tperf_hist, vloss_hist, vperf_hist, num_iter, model) = p8.train_set(trainX_feat, trainY, valX_feat, valY, model, steps, batch_size, loss_fn, optimizer)


(loss, perf) = p8.classify(valX_feat, valY, model, loss_fn)
print "Classification Val performance: ", perf                  #96.67%
(loss, perf) = p8.classify(testX_feat, testY, model, loss_fn)
print "Classification Test performance: ", perf                 #98.33%

p8.draw_curves(tloss_hist,tperf_hist, vloss_hist, vperf_hist, num_iter)