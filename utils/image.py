from torchvision import transforms
import torch
from PIL import Image
from torch import Tensor


def blendTwoImages(img1: Tensor, img2: Tensor, batch_size: int):
    '''
    Blend two images together (grey scale img)
    '''
    img1 = img1.cpu()  # torch.Size([batch, 1, 224, 224])
    img2 = img2.cpu()  # torch.Size([batch, 1, 224, 224])

    t = transforms.ToPILImage()
    img = torch.zeros(batch_size, 1, 224, 224)

    for i in range(batch_size):
        img1_ = t(img1[i])
        img2_ = t(img2[i])
        img1_ = img1_.convert('L')
        img2_ = img2_.convert('L')
        # print(img1_.size) (224, 224)
        img_ = Image.blend(img1_, img2_, 0.3)
        img[i] = transforms.ToTensor()(img_)
    img = torch.cat((img, img, img), dim=1, out=None)
    return img
