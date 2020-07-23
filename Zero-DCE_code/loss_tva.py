import torch
import torchvision
import torch.nn.functional as f

import kornia

import cv2
import numpy as np
import matplotlib.pyplot as plt

from dataloader import SiceDataset

import time

def loss_tva(A):
    """ Calculate the illumination smoothness loss for each curve parameter
    map. This loss preserves the monotonicity relations between neighboring
    pixels.

    Parameters
    ----------
    A: input tensor of shape (m * N, 3, *, *) containing m * N curve parameter
    maps for each m images in the batch.

    Output
    ------
    returns the illumination smoothness loss """

    edges = kornia.filters.SpatialGradient()(A)

    sum = (edges[:, 0, 0, :, :] + edges[:, 0, 1, :, :]) ** 2 + \
            (edges[:, 1, 0, :, :] + edges[:, 1, 1, :, :]) ** 2 + \
            (edges[:, 2, 0, :, :] + edges[:, 2, 1, :, :]) ** 2

    return torch.mean(torch.sum(sum, dim=[1, 2]))

if __name__ == "__main__":

    loss_tva(torch.randint(high=5, size=(6, 3, 4, 4), dtype=torch.float32))
