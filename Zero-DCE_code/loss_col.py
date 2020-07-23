import torch
import torchvision
import torch.nn.functional as f

import cv2
import numpy as np
import matplotlib.pyplot as plt

from dataloader import SiceDataset

import time

def loss_col(Y):
    """ Calculate the color constancy loss over the input batch based on the
    Gray-World color constancy hypothesis that color in each sensor channel
    averages to gray over the entire image.

    Parameters
    ----------
    Y: input tensor of shape (m, 3, *, *) containing the corresponding batch
        of the processed data

    Output
    ------
    returns the color constancy loss """

    Y_averaged = torch.mean(Y, dim=[2, 3])
    print(Y_averaged)
    return (Y_averaged[:, 0] - Y_averaged[:, 1]) ** 2 + \
            (Y_averaged[:, 0] - Y_averaged[:, 2]) ** 2 + \
            (Y_averaged[:, 1] - Y_averaged[:, 2]) ** 2 

