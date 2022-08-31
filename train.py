
import os
import glob
import time
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb

import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm as tqdm


print(torch.__config__.show()) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_colab = None

from utils import *
from dataset import *
from models import *

CUDA_LAUNCH_BLOCKING=1
if use_colab == True:
    path = 'images'
else:
    path = "images"
    
paths = glob.glob(path + "/*.jpg") # Grabbing all the image file names
np.random.seed(123)
paths_subset = np.random.choice(paths, 2500, replace=False) # choosing 1000 images randomly
rand_idxs = np.random.permutation(2500)
train_idxs = rand_idxs[:2000] # choosing the first 8000 as training set
val_idxs = rand_idxs[500:] # choosing last 2000 as validation set
train_paths = paths_subset[train_idxs]
val_paths = paths_subset[val_idxs]
print(len(train_paths), len(val_paths))


train_dl = make_dataloaders(paths=train_paths, split='train')
val_dl = make_dataloaders(paths=val_paths, split='val')

data = next(iter(train_dl))
Ls, abs_ = data['L'], data['ab']
print(Ls.shape, abs_.shape)
print(len(train_dl), len(val_dl))
save_epoch=100

def train_model(model, train_dl, epochs, display_every=10):
    # data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
    num_of_gpus = torch.cuda.device_count()
    model = nn.DataParallel(model,device_ids=[*range(num_of_gpus)])
    model = model.module
    for e in range(epochs):
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to 
        i = 0                                  # log the losses of the complete network
        for data in tqdm(train_dl):
            model.setup_input(data) 
            model.optimize()
            # print('shape of data',data.shape)
            update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            i += 1
        if e % display_every == 0:
            print(f"\nEpoch {e+1}/{epochs}")
            # print(f"Iteration {i}/{len(train_dl)}")
            log_results(loss_meter_dict) # function to print out the losses
            name = f"visualize/colorization_{time.time()}.png"
            # visualize(model, data, save=True,save_name=name) # function displaying the model's outputs
        if e % save_epoch == 0:
            PATH = 'checkpoints/epoch_{}.pt'.format(e)
            torch.save(model.state_dict(), PATH)

def train_model_l1(model, train_dl, epochs, display_every=10):
    data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
    for e in range(epochs):
        loss_meter_dict = create_loss_meters_l1() # function returing a dictionary of objects to 
        i = 0                                  # log the losses of the complete network
        for data in tqdm(train_dl):
            model.setup_input(data) 
            model.optimize_l1()
            # print('shape of data',data.shape)
            update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            i += 1
        if e % display_every == 0:
            print(f"\nEpoch {e+1}/{epochs}")
            # print(f"Iteration {i}/{len(train_dl)}")
            log_results(loss_meter_dict) # function to print out the losses
            name = f"visualize/colorization_{time.time()}.png"
            # visualize(model, data, save=True,save_name=name) # function displaying the model's outputs
        if e% save_epoch == 0:
            PATH = 'checkpoints/epoch_l1_{}.pt'.format(e)
            torch.save(model.state_dict(), PATH)

model = MainModel()
train_model(model, train_dl, 4505)
# train_model_l1(model, train_dl, 505)