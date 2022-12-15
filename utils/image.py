from torchvision import transforms
import torch
from PIL import Image
from torch import Tensor
import cv2
import numpy as np


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

<<<<<<< HEAD
=======

>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
class RandomGaussianBlur(object):
    def __call__(self, img):
        do_it = np.random.rand() > 0.5
        if not do_it:
            return img
        sigma = np.random.rand() * 1.9 + 0.1
        return cv2.GaussianBlur(np.asarray(img), (23, 23), sigma)

<<<<<<< HEAD
=======

>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort
