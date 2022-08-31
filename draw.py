#  python draw.py -checkpoint1='checkpoints1/epoch_l1_500.pt' -checkpoint2='checkpoints/epoch_4500.pt'

from matplotlib import gridspec
import torch
import matplotlib.pyplot as plt
import argparse
import PIL
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")

import glob

from models import MainModel
from utils import lab_to_rgb

def load_model(checkpoint,device):
    model = MainModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=device
        )
    )
    return model

parser = argparse.ArgumentParser(description='model demo')
parser.add_argument('-checkpoint1', type=str, metavar='',help='checkpoint path 1',default='checkpoints1/epoch_l1_500.pt')
parser.add_argument('-checkpoint2', type=str, metavar='',help='checkpoint path 2',default='checkpoints/epoch_4500.pt')
# parser.add_argument('-checkpoint3', type=str, metavar='',required=True,help='checkpoint path 3')
parser.add_argument('-name', type=str, metavar='',default='name for saving plot')
parser.add_argument('-model1-name', type=str, metavar='',help='name for model checkpoint1',default='L1')
parser.add_argument('-model2-name', type=str, metavar='',help='name for model checkpoint2',default='GAN+L1')
# parser.add_argument('-model3-name', type=str, metavar='',help='name for model checkpoint3',default='model 3')

args = parser.parse_args()

path = "images"   
paths = glob.glob(path + "/*.jpg") # Grabbing all the image file names

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model1 = load_model(args.checkpoint1,device)
model2 = load_model(args.checkpoint2,device)
# model3 = load_model(args.checkpoint3)
model1.eval()
model2.eval()


# plot images
fnsize = 17
fig = plt.figure(figsize=(20,10))
k=1
transform = transforms.Compose([transforms.PILToTensor()])

# grid = gridspec.GridSpec(5,3)
# grid.update(wspace=0.000025,hspace=0.05)
for num,image_path in enumerate(paths[27:30]):
# for num,image_path in enumerate(paths[150:153]):
# for num,image_path in enumerate(paths[50:53]):
# for num,image_path in enumerate(paths):
    image = PIL.Image.open(image_path)
    image = image.resize((256, 256))
    gray_image = ImageOps.grayscale(image)
    original_tensor = transform(image).permute(1,2,0) # original color image tensor
    original_tensor = original_tensor.to(torch.float32)/255
    
    gray_tensor = transform(gray_image).permute(1,2,0)
    gray_tensor = gray_tensor.to(torch.float32)/255

    if original_tensor.shape[2]>3:
        pass   
    else: 
        # to make it between -1 and 1
        img = transforms.ToTensor()(image)[:1] * 2. - 1.
        img_input = transforms.ToPILImage()(img)  # l input to model
        with torch.no_grad():
            preds1 = model1.net_G(img.unsqueeze(0).to(device))
            preds2 = model2.net_G(img.unsqueeze(0).to(device))
        colorized1 = lab_to_rgb(img.unsqueeze(0), preds1.cpu())[0] # is numpy array
        colorized2 = lab_to_rgb(img.unsqueeze(0), preds2.cpu())[0] # is numpy array
    
        output1 = torch.from_numpy(colorized1)
        output2 = torch.from_numpy(colorized2)


    ax = fig.add_subplot(3,4, k)
    # ax = plt.subplot(grid[k-1])
    if num==0:
         plt.title('Gray scale input',fontsize=fnsize)
    plt.axis('off')
    plt.imshow(gray_tensor,cmap='gray')
    # ax.set_aspect('equal')

    ax=fig.add_subplot(3,4,k+1)
    # ax = plt.subplot(grid[k])
    if num==0:
        plt.title(str(args.model1_name),fontsize=fnsize)
    plt.axis('off')
    plt.imshow(output1)
    # ax.set_aspect('equal')
   
  
    ax=fig.add_subplot(3,4,k+2)
    # ax = plt.subplot(grid[k+1])
    if num==0:
        plt.title(str(args.model2_name),fontsize=fnsize)
    plt.axis('off')
    plt.imshow(output2)
    # ax.set_aspect('equal')
    

    ax=fig.add_subplot(3,4,k+3)
    # ax = plt.subplot(grid[k+1])
    if num==0:
        plt.title('Ground truth',fontsize=fnsize)
    plt.axis('off')
    plt.imshow(original_tensor)
    # ax.set_aspect('equal')
    k = k+4

  
plt.tight_layout()
plt.subplots_adjust(wspace=0,hspace=0.2)


save_name ='comparison.png'
plt.savefig(save_name)
plt.show()
    

