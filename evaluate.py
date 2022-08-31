
import os
import glob
import time
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
import PIL

import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm as tqdm
 
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


from utils import *
from dataset import *
from models import *
# import cv2

print(torch.__config__.show()) 

CUDA_LAUNCH_BLOCKING=1

path = "images"
    
paths = glob.glob(path + "/*.jpg") # Grabbing all the image file names
# print(paths)

model = MainModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(
    torch.load(
        "checkpoints/epoch_20.pt",
        map_location=device
    )
)
transform = transforms.Compose([transforms.PILToTensor()])

def Average(lst):
   return sum(lst) / len(lst)

mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()

dict_val_model = {
   'l2':[],
   'l1':[],
   'psnr':[],
   'ssim':[],
}
model.eval()
with torch.no_grad():
    for image_path in paths:
        image = PIL.Image.open(image_path)
        # image = cv2.imread(image_path))
        image = image.resize((256, 256))
        # image.show()
        original_tensor = transform(image).permute(2,1,0) # original color image tensor
        original_tensor = original_tensor.to(torch.float32)
        if original_tensor.shape[2]>3:
            pass   
        else: 
            # print(type(original_tensor))
            # print(original_tensor.shape)  
        
            # to make it between -1 and 1
            img = transforms.ToTensor()(image)[:1] * 2. - 1.
            img_input = transforms.ToPILImage()(img)  # l input to model
            # img_input.show()

            model.eval()
            with torch.no_grad():
                preds = model.net_G(img.unsqueeze(0).to(device))
            colorized = lab_to_rgb(img.unsqueeze(0), preds.cpu())[0] # is numpy array
            # print(colorized.shape)
            # print(type(colorized))
            # print(type(original_tensor))
            # print(original_tensor.shape)
            output = torch.from_numpy(colorized)*255.
            # print(output.shape)
            # print(type(output))

            # print(original_tensor.max())
            # print(original_tensor.min())

            # print(output.max())
            # print(output.min())
            
            l1_model = l1_loss(original_tensor,output) 
            mse_model = mse_loss(original_tensor,output) 
            # print(l1_model)
            # print(mse_model)
            
            dict_val_model['l1'].append(l1_model)
            dict_val_model['l2'].append(mse_model)

            # plt.imsave('infer.png', colorized, cmap='Greys')


print('L1 ERROR for Model',Average(dict_val_model['l1']))
print('MSE for MOdel',Average(dict_val_model['l2']))

