import os
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision.transforms import transforms


def get_transforms(img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform


class TrainDataset(data.Dataset):
    """ No label. """

    def __init__(self, args, transform=None):
        self.data_root = 'archive/' + args.data + '/train/'
        self.transform = transform
        self.img_list = os.listdir(os.path.join(self.data_root))

    def load_image(self, path):
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, index):
        image_name = self.img_list[index]
        img_path = os.path.join(self.data_root, image_name)
        img = self.load_image(img_path)

        return img

    def __len__(self):
        return len(self.img_list)
