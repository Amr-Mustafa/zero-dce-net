import torch
import torchvision
import torch.nn.functional as f

import cv2
import numpy as np
import matplotlib.pyplot as plt

from dataloader import SiceDataset

import time

def loss_spa(I, Y, region_size):
    """ Calculate the spatial consistency loss over the entire training set.

    Parameters
    ----------
    I: input tensor of shape (m, 3, I.shape[2], I.shape[2]) containing a batch of the training data
    Y: input tensor of shape (m, 3, I.shape[2], I.shape[2]) containing the corresponding batch of the processed data
    region_size: size of local regions to operate on

    Output
    ------
    returns the spatial consistency loss
    """
   
    # take the average of the 3 color channels
    I_averaged = torch.mean(I, dim=1, keepdim=True)
    Y_averaged = torch.mean(Y, dim=1, keepdim=True)
    
    assert I_averaged.shape == (I.shape[0], 1, I.shape[2], I.shape[2])
    assert Y_averaged.shape == (Y.shape[0], 1, I.shape[2], I.shape[2])

    I_averaged = torch.transpose(I_averaged, 2, 3)
    Y_averaged = torch.transpose(Y_averaged, 2, 3)

    # extract image regions
    I_regions = f.unfold(I_averaged, kernel_size=region_size, stride=region_size)
    Y_regions = f.unfold(Y_averaged, kernel_size=region_size, stride=region_size)
    num_regions_sqrt = I.shape[2] // region_size
    
    assert I_regions.shape == (I.shape[0], 1 * region_size ** 2, num_regions_sqrt ** 2)
    assert Y_regions.shape == (Y.shape[0], 1 * region_size ** 2, num_regions_sqrt ** 2)

    # average each region's intensity
    I_regions_averaged = torch.mean(I_regions, dim=1, keepdim=True)
    Y_regions_averaged = torch.mean(Y_regions, dim=1, keepdim=True)

    assert I_regions_averaged.shape == (I.shape[0], 1, num_regions_sqrt ** 2)
    assert Y_regions_averaged.shape == (Y.shape[0], 1, num_regions_sqrt ** 2)

    I_regions_up = torch.roll(I_regions_averaged, shifts=1, dims=2)
    Y_regions_up = torch.roll(Y_regions_averaged, shifts=1, dims=2)
  
   
    # handle border regions
    I_regions_up[:,:,0:num_regions_sqrt**2:num_regions_sqrt] = 0
    Y_regions_up[:,:,0:num_regions_sqrt**2:num_regions_sqrt] = 0
    
    I_regions_down = torch.roll(I_regions_averaged, shifts=-1, dims=2)
    Y_regions_down = torch.roll(Y_regions_averaged, shifts=-1, dims=2)

     # handle border regions
    I_regions_down[:,:,num_regions_sqrt-1:num_regions_sqrt**2:num_regions_sqrt] = 0
    Y_regions_down[:,:,num_regions_sqrt-1:num_regions_sqrt**2:num_regions_sqrt] = 0

    I_regions_left = torch.roll(I_regions_averaged, shifts=num_regions_sqrt, dims=2)
    Y_regions_left = torch.roll(Y_regions_averaged, shifts=num_regions_sqrt, dims=2)

    # handle border regions
    I_regions_left[:,:,0:num_regions_sqrt] = 0
    Y_regions_left[:,:,0:num_regions_sqrt] = 0

    I_regions_right = torch.roll(I_regions_averaged, shifts=-num_regions_sqrt, dims=2)
    Y_regions_right = torch.roll(Y_regions_averaged, shifts=-num_regions_sqrt, dims=2)

    # handle border regions
    I_regions_right = torch.flip(I_regions_right, dims=(1,2))
    I_regions_right[:,:,0:num_regions_sqrt] = 0
    I_regions_right = torch.flip(I_regions_right, dims=(1,2))

    Y_regions_right = torch.flip(Y_regions_right, dims=(1,2))
    Y_regions_right[:,:,0:num_regions_sqrt] = 0
    Y_regions_right = torch.flip(Y_regions_right, dims=(1,2))
   
    # compute absolute differences in intensities
    I_abs_diff_up = torch.abs(I_regions_averaged - I_regions_up)
    Y_abs_diff_up = torch.abs(Y_regions_averaged - Y_regions_up)
    
    I_abs_diff_down = torch.abs(I_regions_averaged - I_regions_down)
    Y_abs_diff_down = torch.abs(Y_regions_averaged - Y_regions_down)
    
    I_abs_diff_left = torch.abs(I_regions_averaged - I_regions_left)
    Y_abs_diff_left = torch.abs(Y_regions_averaged - Y_regions_left)

    I_abs_diff_right = torch.abs(I_regions_averaged - I_regions_right)
    Y_abs_diff_right = torch.abs(Y_regions_averaged - Y_regions_right)

    up_term = I_abs_diff_up**2 + Y_abs_diff_up**2 - 2*I_abs_diff_up*Y_abs_diff_up
    down_term = I_abs_diff_down**2 + Y_abs_diff_down**2 - 2*I_abs_diff_down*Y_abs_diff_down
    left_term = I_abs_diff_left**2 + Y_abs_diff_left**2 - 2*I_abs_diff_left*Y_abs_diff_left
    right_term = I_abs_diff_right**2 + Y_abs_diff_right**2 - 2*I_abs_diff_right*Y_abs_diff_right

    loss = torch.mean(up_term + down_term + left_term + right_term, dim=2)
    print(loss)

    return loss

def _loss_spa(I, Y, region_size):

    loss = 0.0

    I = torch.mean(I, dim=1, keepdim=True)
    Y = torch.mean(Y, dim=1, keepdim=True)

    for i in range(0, I.shape[2]//region_size):
        for j in range(0, I.shape[2]//region_size):

            I_patch = I[:,:,i*region_size:i*region_size+region_size, j*region_size:j*region_size+region_size]
            I_patch_avg = torch.mean(I_patch, dim=[2,3]).squeeze()

            Y_patch = Y[:,:,i*region_size:i*region_size+region_size, j*region_size:j*region_size+region_size]
            Y_patch_avg = torch.mean(Y_patch, dim=[2,3]).squeeze()
           
            I_patch_up = I[:,:,(i-1)*region_size:(i-1)*region_size+region_size, j*region_size:j*region_size+region_size]
            I_patch_up_avg = torch.mean(I_patch_up, dim=[2,3]).squeeze() if (I_patch_up.shape[2] != 0 and I_patch_up.shape[3] != 0) else 0

            Y_patch_up = Y[:,:,(i-1)*region_size:(i-1)*region_size+region_size, j*region_size:j*region_size+region_size]
            Y_patch_up_avg = torch.mean(Y_patch_up, dim=[2,3]).squeeze() if (Y_patch_up.shape[2] != 0 and Y_patch_up.shape[3] != 0) else 0

            I_patch_down = I[:,:,(i+1)*region_size:(i+1)*region_size+region_size, j*region_size:j*region_size+region_size]
            I_patch_down_avg = torch.mean(I_patch_down, dim=[2,3]).squeeze() if (I_patch_down.shape[2] != 0 and I_patch_down.shape[3] != 0) else 0

            Y_patch_down = Y[:,:,(i+1)*region_size:(i+1)*region_size+region_size, j*region_size:j*region_size+region_size]
            Y_patch_down_avg = torch.mean(Y_patch_down, dim=[2,3]).squeeze() if (Y_patch_down.shape[2] != 0 and Y_patch_down.shape[3] != 0) else 0
 
            I_patch_left = I[:,:,i*region_size:i*region_size+region_size, (j-1)*region_size:(j-1)*region_size+region_size]
            I_patch_left_avg = torch.mean(I_patch_left, dim=[2,3]).squeeze() if (I_patch_left.shape[2] != 0 and I_patch_left.shape[3] != 0) else 0

            Y_patch_left = Y[:,:,i*region_size:i*region_size+region_size, (j-1)*region_size:(j-1)*region_size+region_size]
            Y_patch_left_avg = torch.mean(Y_patch_left, dim=[2,3]).squeeze() if (Y_patch_left.shape[2] != 0 and Y_patch_left.shape[3] != 0) else 0

            I_patch_right = I[:,:,i*region_size:i*region_size+region_size, (j+1)*region_size:(j+1)*region_size+region_size] 
            I_patch_right_avg = torch.mean(I_patch_right, dim=[2,3]).squeeze() if (I_patch_right.shape[2] != 0 and I_patch_right.shape[3] != 0) else 0

            Y_patch_right = Y[:,:,i*region_size:i*region_size+region_size, (j+1)*region_size:(j+1)*region_size+region_size] 
            Y_patch_right_avg = torch.mean(Y_patch_right, dim=[2,3]).squeeze() if (Y_patch_right.shape[2] != 0 and Y_patch_right.shape[3] != 0) else 0

            up_term = (torch.abs(Y_patch_avg - Y_patch_up_avg) - torch.abs(I_patch_avg - I_patch_up_avg))**2
            down_term =  (torch.abs(Y_patch_avg - Y_patch_down_avg) - torch.abs(I_patch_avg - I_patch_down_avg))**2
            left_term = (torch.abs(Y_patch_avg - Y_patch_left_avg) - torch.abs(I_patch_avg - I_patch_left_avg))**2
            right_term = (torch.abs(Y_patch_avg - Y_patch_right_avg) - torch.abs(I_patch_avg - I_patch_right_avg))**2
       
            loss += up_term + down_term + left_term + right_term

    loss /= (I.shape[2]//region_size)**2
    print(loss.squeeze())

if __name__ == "__main__":

    ## prepare the dataset and the loaders
    dataset = SiceDataset("/home/amrmustafa/vault/Zero-DCE/Dataset_Part1/[0-9]*/") 
    train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            shuffle=True,
            num_workers=0
        )

    first_batch = next(iter(train_loader))
    second_batch = next(iter(train_loader))
    
    ## visualize a single training batch
    grid = torchvision.utils.make_grid(first_batch, nrow=2)
    plt.figure(figsize=(50, 50))
    #plt.imshow(np.transpose(grid, (1,2,0)))
    #plt.show()

    start_time = time.time()
    _loss_spa(first_batch, second_batch, 64)
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    loss_spa(first_batch, second_batch, 64)
    print("--- %s seconds ---" % (time.time() - start_time))





    


