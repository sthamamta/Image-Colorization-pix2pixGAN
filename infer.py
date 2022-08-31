
import PIL
import torch
from matplotlib import pyplot as plt
from torchvision import transforms

from models import MainModel
from utils import lab_to_rgb

if __name__ == '__main__':
    model = MainModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(
        torch.load(
            "checkpoints/epoch_l1_20.pt",
            map_location=device
        )
    )
    path = "images/sun_aacfwyqmoevotupe.jpg"
    img = PIL.Image.open(path)
    img = img.resize((256, 256))
    # img.show()
    # change range to -1 and 1
    img = transforms.ToTensor()(img)[:1] * 2. - 1.
    img_input = transforms.ToPILImage()(img)
    # img_input.show()
    # print(img_input.shape)
    model.eval()
    with torch.no_grad():
        preds = model.net_G(img.unsqueeze(0).to(device))
    colorized = lab_to_rgb(img.unsqueeze(0), preds.cpu())[0]
    # print(colorized.shape)
    # print(type(colorized))
    # plt.savefig('infer.png')
    plt.imsave('infer.png', colorized, cmap='Greys')
    # plt.imshow(colorized)
