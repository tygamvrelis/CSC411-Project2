from pylab import *
import numpy as np
import torch
from torch.autograd import Variable

def get_feature_set(X,size, RESOLUTION, modelAN):
    # Read an image
    first_pass = True
    for ind in range(size):
        x = X[ind * RESOLUTION:(ind + 1) * RESOLUTION, :]
        im = x[:,:,:3]
        im = im - np.mean(im.flatten())

        im = im/np.max(np.abs(im.flatten()))

        im = np.rollaxis(im, -1).astype(np.float32)
        # Turn the image into a numpy variable

        im_v = Variable(torch.from_numpy(im).unsqueeze_(0), requires_grad=False)
        if torch.cuda.is_available():
            im_v = im_v.cuda()
        res = modelAN.forward(im_v)
        if torch.cuda.is_available():
            res = res.cpu()
        res = res.data.numpy()
        if first_pass:
            batch_xs = np.zeros((0, res.shape[1]))
            first_pass = False
        batch_xs = np.vstack((batch_xs, res))
    return batch_xs