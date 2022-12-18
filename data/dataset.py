import os
import numpy as np
from PIL import Image
import torch
import random
from torch.utils import data
from torchvision.transforms import transforms
import pandas as pd
from torch.utils.data import DataLoader

from utils.info import terminal_msg
from utils.image import RandomGaussianBlur, get_color_distortion

from data.sampler import ImbalancedDatasetSampler, MultilabelBalancedRandomSampler

def get_data_weights(args):
    num = {}
    data_dict = args.data.split(", ") # ['ODIR-5K', 'TAOP', 'RFMiD', ...]
    weights = []

    if 'ODIR-5K' in args.data:
        num['ODIR-5K'] = 6037
    if 'RFMiD' in args.data:
        num['RFMiD'] = 1920
    if 'TAOP' in args.data:
        num['TAOP'] = 3000
    if 'APTOS' in args.data:
        num['APTOS'] = 3295
    if 'Kaggle' in args.data:
        num['Kaggle'] = 35126
    if 'DR+' in args.data:
        num['DR+'] = 51491
    for i in data_dict:
        weights.append(10000/num[i])
    weights = np.array(weights, dtype=np.float32)

    return weights

def get_train_transforms(img_size):
    '''
    resize (256)-> random crop (224) -> random filp -> color distortion -> GaussianBlur -> normalize
    '''
    transform = transforms.Compose([
        #transforms.Resize((256, 256)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Compose([
            get_color_distortion(),
            RandomGaussianBlur(),
            ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def get_valid_transforms(img_size):
    '''
    resize (224)-> normalize
    '''
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    if "APTOS" in args.data:
        datasets["APTOS"] = TrainDataset('APTOS', transform)
    if "Kaggle" in args.data:
        datasets["Kaggle"] = TrainDataset('Kaggle', transform)
    if "DR+" in args.data:
        datasets["DR+"] = TrainDataset('DR+', transform)
    else:
        terminal_msg("Args.Data Error (From get_train_datasets)", "F")
        exit()
    return datasets

def get_valid_datasets(args, transform):
    datasets = {}
    if "ODIR-5K" in args.data:
        datasets["ODIR-5K"] = ValidDataset('ODIR-5K', transform)
    if "RFMiD" in args.data:
        datasets["RFMiD"] = ValidDataset('RFMiD', transform)
    if "TAOP" in args.data:
        datasets["TAOP"] = ValidDataset('TAOP', transform)
    if "APTOS" in args.data:
        datasets["APTOS"] = ValidDataset('APTOS', transform)
    if "Kaggle" in args.data:
        datasets["Kaggle"] = ValidDataset('Kaggle', transform)
    if "DR+" in args.data:
        datasets["DR+"] = ValidDataset('DR+', transform)
    else:
        terminal_msg("Args.Data Error (From get_valid_datasets)", "F")
        exit()
    return datasets

def get_train_dataloader(args, transform):
    dataloaders = {}
    if "ODIR-5K" in args.data:
        dataloaders["ODIR-5K"] = DataLoader(TrainDataset('ODIR-5K', transform), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    if "RFMiD" in args.data:
        dataloaders["RFMiD"] = DataLoader(TrainDataset('RFMiD', transform), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    if "TAOP" in args.data:
        dataloaders["TAOP"] = DataLoader(TrainDataset('TAOP', transform), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    if "APTOS" in args.data:
        dataloaders["APTOS"] = DataLoader(TrainDataset('APTOS', transform), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    if "Kaggle" in args.data:
        dataloaders["Kaggle"] = DataLoader(TrainDataset('Kaggle', transform), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    if "DR+" in args.data:
        dataloaders["DR+"] = DataLoader(TrainDataset('DR+', transform), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    else:
        terminal_msg("Args.Data Error (From get_train_dataloder)", "F")
        exit()
    return dataloaders

def get_valid_dataloader(args, transform):
    dataloaders = {}
    if "ODIR-5K" in args.data:
        dataloaders["ODIR-5K"] = DataLoader(ValidDataset('ODIR-5K', transform), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    if "RFMiD" in args.data:
        dataloaders["RFMiD"] = DataLoader(ValidDataset('RFMiD', transform), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    if "TAOP" in args.data:
        dataloaders["TAOP"] = DataLoader(ValidDataset('TAOP', transform), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    if "APTOS" in args.data:
        dataloaders["APTOS"] = DataLoader(ValidDataset('APTOS', transform), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    if "Kaggle" in args.data:
        dataloaders["Kaggle"] = DataLoader(ValidDataset('Kaggle', transform), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    if "DR+" in args.data:
        dataloaders["DR+"] = DataLoader(ValidDataset('DR+', transform), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    assert dataloaders
    return dataloaders

def get_train_data(args, transform):
    data = {}

    if 'ODIR-5K' in args.data:
        data['ODIR-5K'] = {}
        train_dataset = TrainDataset('ODIR-5K', transform)
        if args.balanced_sampling:
            data['ODIR-5K']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, sampler=MultilabelBalancedRandomSampler(train_dataset.get_labels()), pin_memory= True, num_workers=args.num_workers)
        else:
            data['ODIR-5K']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
        data['ODIR-5K']['iterloader'] = iter(data['ODIR-5K']['dataloader'])
    
    if 'RFMiD' in args.data:
        data['RFMiD'] = {}
        train_dataset = TrainDataset('RFMiD', transform)
        if args.balanced_sampling:
            data['RFMiD']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, sampler=MultilabelBalancedRandomSampler(train_dataset.get_labels()), pin_memory= True, num_workers=args.num_workers)
        else:
            data['RFMiD']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
        data['RFMiD']['iterloader'] = iter(data['RFMiD']['dataloader'])
    
    if 'DR+' in args.data:
        data['DR+'] = {}
        data['DR+'] = {}
        train_dataset = TrainDataset('DR+', transform)
        if args.balanced_sampling:
            data['DR+']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, sampler=MultilabelBalancedRandomSampler(train_dataset.get_labels()), pin_memory= True, num_workers=args.num_workers)
        else:
            data['DR+']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
        data['DR+']['iterloader'] = iter(data['DR+']['dataloader'])
    
    if 'TAOP' in args.data:
        data['TAOP'] = {}
        train_dataset = TrainDataset('TAOP', transform)
        if args.balanced_sampling:
            data['TAOP']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, sampler=ImbalancedDatasetSampler(train_dataset), pin_memory= True, num_workers=args.num_workers)
        else:
            data['TAOP']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
        data['TAOP']['iterloader'] = iter(data['TAOP']['dataloader'])
    
    if 'APTOS' in args.data:
        data['APTOS'] = {}
        train_dataset = TrainDataset('APTOS', transform)
        if args.balanced_sampling:
            data['APTOS']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, sampler=ImbalancedDatasetSampler(train_dataset), pin_memory= True, num_workers=args.num_workers)
        else:
            data['APTOS']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
        data['APTOS']['iterloader'] = iter(data['APTOS']['dataloader'])
    
    if 'Kaggle' in args.data:
        data['Kaggle'] = {}
        train_dataset = TrainDataset('Kaggle', transform)
        if args.balanced_sampling:
            data['Kaggle']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, sampler=ImbalancedDatasetSampler(train_dataset), pin_memory= True, num_workers=args.num_workers)
        else:
            data['Kaggle']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
        data['Kaggle']['iterloader'] = iter(data['Kaggle']['dataloader'])
    
    assert data
    return data

def get_batch(data):
    try:
        batch = next(data['iterloader'])
    except StopIteration:
        data['iterloader'] = iter(data['dataloader'])
        batch = next(data['iterloader'])
    return batch

def get_single_task_train_dataloader(args, train_transfrom, valid_transform):
    train_dataset = TrainDataset(args.data, transform=train_transfrom)
    valid_dataset = ValidDataset(args.data, transform=valid_transform)

    if args.balanced_sampling:
        if args.data in ["ODIR-5K", "RFMiD", "DR+"]:
            sampler = MultilabelBalancedRandomSampler(train_dataset.get_labels())
        elif args.data in ["TAOP", "APTOS", "Kaggle"]:
            sampler = ImbalancedDatasetSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

    return train_dataloader, valid_dataloader


#=====================Datasets=========================#

class TrainDataset(data.Dataset):
    """ 
    Based on the args.data to choose the dataset for training:

    ODIR-5K: 6,307 samples
    RFMiD: 1,920 samples
    DR+: 51,491 samples

    TAOP: 3,000 samples
    APTOS: 3,295 samples
    Kaggle: 35,126 samples
    """

    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data

        if self.data == 'ODIR-5K':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/ODIR-5K/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_train.csv')
        elif self.data == 'RFMiD':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/RFMiD/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_train.csv')
        elif self.data == 'TAOP':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/TAOP-2021/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_train.csv')
        elif self.data == 'APTOS':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/APTOS/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_train.csv')
        elif self.data == 'Kaggle':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/Kaggle/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'trainLabels.csv')
        elif self.data == 'DR+':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/KaggleDR+/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_train.csv')
        else:
            terminal_msg("Args.Data ({}) Error (From TrainDataset.__init__)".format(data), "F")
            exit()
                
    def load_image(self, path):
        image = Image.open(path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.data == 'ODIR-5K':
            img_path = os.path.join(self.data_root, 'train_resized/', self.landmarks_frame.iloc[idx, 0] + '.jpg')
        elif self.data == 'RFMiD':
            img_path = os.path.join(self.data_root, 'train_resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        elif self.data == 'TAOP':
            img_path = os.path.join(self.data_root, 'png_resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        elif self.data == 'APTOS':
            img_path = os.path.join(self.data_root, 'train_resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        elif self.data == 'Kaggle':
            img_path = os.path.join(self.data_root, 'train_resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpeg')
        elif self.data == 'DR+':
            index = str(self.landmarks_frame.iloc[idx, 0])
            if os.path.exists(f'/mnt/data3_ssd/RetinalDataset/Kaggle/train_resized/{index}'):
                img_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/train_resized/', index)
            elif os.path.exists(f'/mnt/data3_ssd/RetinalDataset/Kaggle/valid_resized/{index}'):
                img_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/valid_resized/', index)
            else:
                print('Cannot find img path (DR+)')
                exit()
        else:
            terminal_msg("Args.Data Error (From TrainDataset.__getitem__)", "F")
            exit()

        img = self.load_image(img_path)
        landmarks = np.array(self.landmarks_frame.iloc[idx, 1:], dtype=np.float32).tolist()

        if self.data in ["ODIR-5K", "RFMiD", "DR+"]:
            sample = {'image': img, 'landmarks': torch.tensor(landmarks).float()}
        elif self.data in ["TAOP", "APTOS", "Kaggle"]:
            sample = {'image': img, 'landmarks': torch.tensor(landmarks).int()}
        else:
            terminal_msg("Args.Data Error (From TrainDataset.__getitem__)", "F")
            exit()

        return sample

    def __len__(self):
       return len(self.landmarks_frame)

    def get_labels(self):
        return self.landmarks_frame.iloc[:,1:]

class ValidDataset(data.Dataset):
    """ 
    Based on the args.data to choose the dataset for validation:

    ODIR-5K: 693 samples
    RFMiD: 640 samples
    Kaggle: 5,721 samples

    TAOP: 297 samples
    APTOS: 367 samples
    Kaggle: 2,000 samples (for valid only)
    """

    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data

        if self.data == 'ODIR-5K':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/ODIR-5K/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_valid.csv')
        elif self.data == 'RFMiD':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/RFMiD/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_valid.csv')
        elif self.data == 'TAOP':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/TAOP-2021/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_valid.csv')
        elif self.data == 'APTOS':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/APTOS/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_valid.csv')
        elif self.data == 'Kaggle':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/Kaggle/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_test.csv')
        elif self.data == 'DR+':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/KaggleDR+/'
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
            img_path = os.path.join(self.data_root, 'valid_resized/', self.landmarks_frame.iloc[idx, 0] + '.jpg')
        elif self.data == 'RFMiD':
            img_path = os.path.join(self.data_root, 'valid_resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        elif self.data == 'TAOP':
            img_path = os.path.join(self.data_root, 'png_resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        elif self.data == 'APTOS':
            img_path = os.path.join(self.data_root, 'train_resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        elif self.data == 'Kaggle':
            img_path = os.path.join(self.data_root, 'valid_resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpeg')
        elif self.data == 'DR+':
            index = str(self.landmarks_frame.iloc[idx, 0])
            if os.path.exists(f'/mnt/data3_ssd/RetinalDataset/Kaggle/train_resized/{index}'):
                img_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/train_resized/', index)
            elif os.path.exists(f'/mnt/data3_ssd/RetinalDataset/Kaggle/valid_resized/{index}'):
                img_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/valid_resized/', index)
            else:
                print('Cannot find img path (DR+)')
                exit()
        else:
            terminal_msg("Args.Data Error (From ValidDataset.__getitem__)", "F")
            exit()

        img = self.load_image(img_path)
        landmarks = np.array(self.landmarks_frame.iloc[idx, 1:], dtype=np.float32).tolist()
        sample = {'image': img, 'landmarks': torch.tensor(landmarks).float()}

        return sample

    def __len__(self):
       return len(self.landmarks_frame)
