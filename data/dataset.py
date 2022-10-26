import os
import numpy as np
from PIL import Image
import torch
from torch.utils import data
from torchvision.transforms import transforms
import pandas as pd
from utils.info import terminal_msg


def get_transforms(img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform


class TrainDataset(data.Dataset):
    """ 
    Based on the args.data to choose the dataset for training:
    ODIR-5k: 6037 samples
    RFMiD: 1920 samples
    TAOP: 3000 samples
    """

    def __init__(self, args, transform=None):
        self.transform = transform
        self.args = args
        # self.img_list = os.listdir(os.path.join(self.data_root))
        if args.data == 'ODIR-5K':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/ODIR-5K/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_train.csv')
        elif args.data == 'RFMiD':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/RFMiD/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'RFMiD_Training_Labels.csv')
        elif args.data == 'TAOP':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/TAOP-2021/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_train.csv')
        else:
            terminal_msg("Args.Data Error (From TrainDataset.__init__)", "F")
            exit()

    def load_image(self, path):
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.args.data == 'ODIR-5K':
            img_path = os.path.join(self.data_root, 'train/', self.landmarks_frame.iloc[idx, 0] + '.jpg')
        elif self.args.data == 'RFMiD':
            img_path = os.path.join(self.data_root, 'train/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        elif self.args.data == 'TAOP':
            img_path = os.path.join(self.data_root, 'png/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        else:
            terminal_msg("Args.Data Error (From TrainDataset.__getitem__)", "F")
            exit()
        img = self.load_image(img_path)
        landmarks = np.array(self.landmarks_frame.iloc[idx, 1:]).tolist()
        if self.args.data in ["ODIR-5K", "RFMiD"]:
            sample = {'image': img, 'landmarks': torch.tensor(landmarks).float()}
        elif self.args.data in ["TAOP"]:
            sample = {'image': img, 'landmarks': torch.tensor(landmarks).int()}

        return sample

    def __len__(self):
        return len(self.landmarks_frame)


class ValidDataset(data.Dataset):
    """ 
    Based on the args.data to choose the dataset for validation:
    ODIR-5K: 693 samples
    RFMiD: 640 samples
    TAOP: 297 samples
    """

    def __init__(self, args, transform=None):
        self.transform = transform
        self.args = args
        # self.img_list = os.listdir(os.path.join(self.data_root))
        if args.data == 'ODIR-5K':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/ODIR-5K/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_valid.csv')
        elif args.data == 'RFMiD':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/RFMiD/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'RFMiD_Validation_Labels.csv')
        elif args.data == 'TAOP':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/TAOP-2021/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_valid.csv')
        else:
            terminal_msg("Args.Data Error (From ValidDataset.__init__)", "F")
            exit()

    def load_image(self, path):
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.args.data == 'ODIR-5K':
            img_path = os.path.join(self.data_root, 'valid/', self.landmarks_frame.iloc[idx, 0] + '.jpg')
        elif self.args.data == 'RFMiD':
            img_path = os.path.join(self.data_root, 'valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        elif self.args.data == 'TAOP':
            img_path = os.path.join(self.data_root, 'png/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        else:
            terminal_msg("Args.Data Error (From ValidDataset.__getitem__)", "F")
            exit()
        img = self.load_image(img_path)
        landmarks = np.array(self.landmarks_frame.iloc[idx, 1:]).tolist()
        sample = {'image': img, 'landmarks': torch.tensor(landmarks).float()}

        return sample

    def __len__(self):
        return len(self.landmarks_frame)
