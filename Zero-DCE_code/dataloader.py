import os
import sys

import torch
import torchvision
import torch.utils.data as data
from torch.nn import functional as f

import numpy as np
from PIL import Image
import glob
import random
import cv2

from sklearn.feature_extraction import image

import matplotlib.pyplot as plt

random.seed(1143)

def populate_train_list(lowlight_images_path):

    # the glob module finds all pathnames matching a specified pattern
    train_list = glob.glob(lowlight_images_path + "*.JPG", recursive=True)
    random.shuffle(train_list)
        
    # the list should contain the paths to all training images
    return train_list

# PyTorch data manipulation pipeline:
# The raw data is modeled by a torch.utils.data.Dataset object which provides a uniform interface to access the data. The dataset can then be used by a torch.utils.data.DataLoader object to provide batches of data during the training loop.

class SiceDataset(data.Dataset):

    def __init__(self, lowlight_images_path):

        # get all pathnames for training images
        self.train_list = populate_train_list(lowlight_images_path) 
            
        # must reshape images to 256x256
        self.size = 256

        self.data_list = self.train_list
        print("Total training examples:", len(self.train_list))

    def __getitem__(self, index):

        # get the image path and load it 
        data_lowlight_path = self.data_list[index]
        data_lowlight = Image.open(data_lowlight_path)

        # resize and normalize the image
        data_lowlight = data_lowlight.resize((self.size,self.size), Image.ANTIALIAS)
        data_lowlight = (np.asarray(data_lowlight)/255.0) 
        data_lowlight = torch.from_numpy(data_lowlight).float()
                
        # change the order of axes from (H,W,C) to (C,H,W) because
        # PyTorch modules dealing with image data require tensors to
        # be laid out as (C,H,W)
        return data_lowlight.permute(2,0,1)

    def __len__(self):
        return len(self.data_list)

if __name__ == "__main__":

    ## prepare the dataset and the loaders
    dataset = SiceDataset("/home/amrmustafa/vault/Zero-DCE/Dataset_Part1/[0-9]*/") 
    train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=10,
            shuffle=True,
            num_workers=2
        )
    batch = next(iter(train_loader))
    
    ## visualize a single training batch
    grid = torchvision.utils.make_grid(batch, nrow=5)
    plt.figure(figsize=(50, 50))
    plt.imshow(np.transpose(grid, (1,2,0)))
    plt.show()

    #loss_spa(batch, batch, 32)

    #image = next(iter(train_loader))
    #print(image.shape)
    #patches = f.unfold(image, kernel_size=32, stride=32)
    #print(patches.shape)
    #im = patches[0].reshape((1, 3, 32, 32, 64)).permute(0, 3, 2, 1, 4)
    #print(im.shape)
