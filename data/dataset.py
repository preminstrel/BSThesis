import os
import numpy as np
from PIL import Image
import torch
from torch.utils import data
from torchvision.transforms import transforms
import pandas as pd
from torch.utils.data import DataLoader
from utils.info import terminal_msg


def get_transforms(img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform


def get_train_datasets(args, transform):
    datasets = {}
    if "ODIR-5K" in args.data:
        datasets["ODIR-5K"] = TrainDataset('ODIR-5K', transform)
    if "RFMiD" in args.data:
        datasets["RFMiD"] = TrainDataset('RFMiD', transform)
    if "TAOP" in args.data:
        datasets["TAOP"] = TrainDataset('TAOP', transform)
    else:
        terminal_msg("Args.Data Error (From get_train_datasets)", "F")
        exit()
    return datasets


def get_train_dataloader(args, transform):
    dataloaders = {}
    if "ODIR-5K" in args.data:
        dataloaders["ODIR-5K"] = DataLoader(TrainDataset('ODIR-5K', transform), batch_size=args.batch_size, shuffle=True)
    if "RFMiD" in args.data:
        dataloaders["RFMiD"] = DataLoader(TrainDataset('RFMiD', transform), batch_size=args.batch_size, shuffle=True)
    if "TAOP" in args.data:
        dataloaders["TAOP"] = DataLoader(TrainDataset('TAOP', transform), batch_size=args.batch_size, shuffle=True)
    else:
        terminal_msg("Args.Data Error (From get_train_dataloder)", "F")
        exit()
    return dataloaders


def get_valid_datasets(args, transform):
    datasets = {}
    if "ODIR-5K" in args.data:
        datasets["ODIR-5K"] = ValidDataset('ODIR-5K', transform)
    if "RFMiD" in args.data:
        datasets["RFMiD"] = ValidDataset('RFMiD', transform)
    if "TAOP" in args.data:
        datasets["TAOP"] = ValidDataset('TAOP', transform)
    else:
        terminal_msg("Args.Data Error (From get_valid_datasets)", "F")
        exit()
    return datasets


def get_valid_dataloader(args, transform):
    dataloaders = {}
    if "ODIR-5K" in args.data:
        dataloaders["ODIR-5K"] = DataLoader(ValidDataset('ODIR-5K', transform), batch_size=128, shuffle=True)
    if "RFMiD" in args.data:
        dataloaders["RFMiD"] = DataLoader(ValidDataset('RFMiD', transform), batch_size=128, shuffle=True)
    if "TAOP" in args.data:
        dataloaders["TAOP"] = DataLoader(ValidDataset('TAOP', transform), batch_size=128, shuffle=True)
    else:
        terminal_msg("Args.Data Error (From get_valid_dataloder)", "F")
        exit()
    return dataloaders


class TrainDataset(data.Dataset):
    """ 
    Based on the args.data to choose the dataset for training:
    ODIR-5k: 6037 samples
    RFMiD: 1920 samples
    TAOP: 3000 samples
    """

    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data
        if self.data == 'ODIR-5K':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/ODIR-5K/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_train.csv')
        elif self.data == 'RFMiD':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/RFMiD/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'RFMiD_Training_Labels.csv')
        elif self.data == 'TAOP':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/TAOP-2021/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_train.csv')
        else:
            terminal_msg("Args.Data ({}) Error (From TrainDataset.__init__)".format(data), "F")
            exit()

    def load_image(self, path):
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.data == 'ODIR-5K':
            img_path = os.path.join(self.data_root, 'train/', self.landmarks_frame.iloc[idx, 0] + '.jpg')
        elif self.data == 'RFMiD':
            img_path = os.path.join(self.data_root, 'train/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        elif self.data == 'TAOP':
            img_path = os.path.join(self.data_root, 'png/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        else:
            terminal_msg("Args.Data Error (From TrainDataset.__getitem__)", "F")
            exit()
        img = self.load_image(img_path)
        landmarks = np.array(self.landmarks_frame.iloc[idx, 1:]).tolist()
        if self.data in ["ODIR-5K", "RFMiD"]:
            sample = {'image': img, 'landmarks': torch.tensor(landmarks).float()}
        elif self.data in ["TAOP"]:
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

    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data
        # self.img_list = os.listdir(os.path.join(self.data_root))
        if self.data == 'ODIR-5K':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/ODIR-5K/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_valid.csv')
        elif self.data == 'RFMiD':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/RFMiD/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'RFMiD_Validation_Labels.csv')
        elif self.data == 'TAOP':
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
        if self.data == 'ODIR-5K':
            img_path = os.path.join(self.data_root, 'valid/', self.landmarks_frame.iloc[idx, 0] + '.jpg')
        elif self.data == 'RFMiD':
            img_path = os.path.join(self.data_root, 'valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        elif self.data == 'TAOP':
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
