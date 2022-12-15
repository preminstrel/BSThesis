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

<<<<<<< HEAD
def get_data_weights(args):
    num = {}
    data_dict = args.data.split(", ") # ['ODIR-5K', 'TAOP', 'RFMiD', ...]
=======

def get_data_weights(args):
    num = {}
    data_dict = args.data.split(", ")  # ['ODIR-5K', 'TAOP', 'RFMiD', ...]
>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
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
<<<<<<< HEAD
        num['Kaggle'] = 35126
    if 'KaggleDR+' in args.data:
        num['KaggleDR+'] = 51491
=======
        num['Kaggle'] = 31613
>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
    for i in data_dict:
        weights.append(10000/num[i])
    weights = np.array(weights, dtype=np.float32)

    return weights

<<<<<<< HEAD
=======

>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
def get_train_transforms(img_size):
    '''
    resize(256)-> random crop -> random filp -> color distortion -> GaussianBlur -> normalize
    '''
    transform = transforms.Compose([
<<<<<<< HEAD
        #transforms.Resize((256, 256)),
=======
        transforms.Resize((256, 256)),
>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Compose([
            get_color_distortion(),
            RandomGaussianBlur(),
<<<<<<< HEAD
            ]),
=======
        ]),
>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

<<<<<<< HEAD
=======

>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
def get_valid_transforms(img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

<<<<<<< HEAD
=======

>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
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
<<<<<<< HEAD
    if "KaggleDR+" in args.data:
        datasets["KaggleDR+"] = TrainDataset('KaggleDR+', transform)
=======
>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
    else:
        terminal_msg("Args.Data Error (From get_train_datasets)", "F")
        exit()
    return datasets

<<<<<<< HEAD
=======

>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
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
<<<<<<< HEAD
    if "KaggleDR+" in args.data:
        datasets["KaggleDR+"] = ValidDataset('KaggleDR+', transform)
=======
>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
    else:
        terminal_msg("Args.Data Error (From get_valid_datasets)", "F")
        exit()
    return datasets

<<<<<<< HEAD
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
    if "KaggleDR+" in args.data:
        dataloaders["KaggleDR+"] = DataLoader(TrainDataset('KaggleDR+', transform), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
=======

def get_train_dataloader(args, transform):
    dataloaders = {}
    if "ODIR-5K" in args.data:
        dataloaders["ODIR-5K"] = DataLoader(TrainDataset('ODIR-5K', transform), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    if "RFMiD" in args.data:
        dataloaders["RFMiD"] = DataLoader(TrainDataset('RFMiD', transform), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    if "TAOP" in args.data:
        dataloaders["TAOP"] = DataLoader(TrainDataset('TAOP', transform), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    if "APTOS" in args.data:
        dataloaders["APTOS"] = DataLoader(TrainDataset('APTOS', transform), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    if "Kaggle" in args.data:
        dataloaders["Kaggle"] = DataLoader(TrainDataset('Kaggle', transform), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
    else:
        terminal_msg("Args.Data Error (From get_train_dataloder)", "F")
        exit()
    return dataloaders

<<<<<<< HEAD
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
    if "KaggleDR+" in args.data:
        dataloaders["KaggleDR+"] = DataLoader(ValidDataset('KaggleDR+', transform), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    assert dataloaders
    return dataloaders

=======

def get_valid_dataloader(args, transform):
    dataloaders = {}
    if "ODIR-5K" in args.data:
        dataloaders["ODIR-5K"] = DataLoader(ValidDataset('ODIR-5K', transform), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    if "RFMiD" in args.data:
        dataloaders["RFMiD"] = DataLoader(ValidDataset('RFMiD', transform), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    if "TAOP" in args.data:
        dataloaders["TAOP"] = DataLoader(ValidDataset('TAOP', transform), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    if "APTOS" in args.data:
        dataloaders["APTOS"] = DataLoader(ValidDataset('APTOS', transform), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    if "Kaggle" in args.data:
        dataloaders["Kaggle"] = DataLoader(ValidDataset('Kaggle', transform), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    assert dataloaders
    return dataloaders


>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
def get_train_data(args, transform):
    data = {}

    if 'ODIR-5K' in args.data:
        data['ODIR-5K'] = {}
<<<<<<< HEAD
        data['ODIR-5K']['dataloader'] = DataLoader(TrainDataset('ODIR-5K', transform), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
        data['ODIR-5K']['iterloader'] = iter(data['ODIR-5K']['dataloader'])
    if 'RFMiD' in args.data:
        data['RFMiD'] = {}
        data['RFMiD']['dataloader'] = DataLoader(TrainDataset('RFMiD', transform), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
        data['RFMiD']['iterloader'] = iter(data['RFMiD']['dataloader'])
    if 'TAOP' in args.data:
        data['TAOP'] = {}
        data['TAOP']['dataloader'] = DataLoader(TrainDataset('TAOP', transform), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
        data['TAOP']['iterloader'] = iter(data['TAOP']['dataloader'])
    if 'APTOS' in args.data:
        data['APTOS'] = {}
        data['APTOS']['dataloader'] = DataLoader(TrainDataset('APTOS', transform), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
        data['APTOS']['iterloader'] = iter(data['APTOS']['dataloader'])
    if 'Kaggle' in args.data:
        data['Kaggle'] = {}
        data['Kaggle']['dataloader'] = DataLoader(TrainDataset('Kaggle', transform), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
        data['Kaggle']['iterloader'] = iter(data['Kaggle']['dataloader'])
    if 'KaggleDR+' in args.data:
        data['KaggleDR+'] = {}
        data['KaggleDR+']['dataloader'] = DataLoader(TrainDataset('KaggleDR+', transform), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
        data['KaggleDR+']['iterloader'] = iter(data['KaggleDR+']['dataloader'])
    
    assert data
    return data

=======
        data['ODIR-5K']['dataloader'] = DataLoader(TrainDataset('ODIR-5K', transform), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
        data['ODIR-5K']['iterloader'] = iter(data['ODIR-5K']['dataloader'])
    if 'RFMiD' in args.data:
        data['RFMiD'] = {}
        data['RFMiD']['dataloader'] = DataLoader(TrainDataset('RFMiD', transform), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
        data['RFMiD']['iterloader'] = iter(data['RFMiD']['dataloader'])
    if 'TAOP' in args.data:
        data['TAOP'] = {}
        data['TAOP']['dataloader'] = DataLoader(TrainDataset('TAOP', transform), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
        data['TAOP']['iterloader'] = iter(data['TAOP']['dataloader'])
    if 'APTOS' in args.data:
        data['APTOS'] = {}
        data['APTOS']['dataloader'] = DataLoader(TrainDataset('APTOS', transform), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
        data['APTOS']['iterloader'] = iter(data['APTOS']['dataloader'])
    if 'Kaggle' in args.data:
        data['Kaggle'] = {}
        data['Kaggle']['dataloader'] = DataLoader(TrainDataset('Kaggle', transform), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
        data['Kaggle']['iterloader'] = iter(data['Kaggle']['dataloader'])

    assert data
    return data


>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
def get_batch(data):
    try:
        batch = next(data['iterloader'])
    except StopIteration:
<<<<<<< HEAD
=======
        print('Loop')
>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
        data['iterloader'] = iter(data['dataloader'])
        batch = next(data['iterloader'])
    return batch

<<<<<<< HEAD
=======

>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
class TrainDataset(data.Dataset):
    """ 
    Based on the args.data to choose the dataset for training:

<<<<<<< HEAD
    ODIR-5K: 6,307 samples
    RFMiD: 1,920 samples
    KaggleDR+: 51,491 samples

    TAOP: 3,000 samples
    APTOS: 3,295 samples
    Kaggle: 35,126 samples
=======
    ODIR-5K: 6,037 samples
    RFMiD: 1,920 samples

    TAOP: 3,000 samples
    APTOS: 3,295 samples
    Kaggle: 31,613 samples
>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
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
<<<<<<< HEAD
            self.landmarks_frame = pd.read_csv(self.data_root + 'trainLabels.csv')
        elif self.data == 'KaggleDR+':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/KaggleDR+/'
=======
>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_train.csv')
        else:
            terminal_msg("Args.Data ({}) Error (From TrainDataset.__init__)".format(data), "F")
            exit()
<<<<<<< HEAD
                
=======

>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
    def load_image(self, path):
        image = Image.open(path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.data == 'ODIR-5K':
<<<<<<< HEAD
            img_path = os.path.join(self.data_root, 'train_resized/', self.landmarks_frame.iloc[idx, 0] + '.jpg')
        elif self.data == 'RFMiD':
            img_path = os.path.join(self.data_root, 'train_resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        elif self.data == 'TAOP':
            img_path = os.path.join(self.data_root, 'png_resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        elif self.data == 'APTOS':
            img_path = os.path.join(self.data_root, 'train_resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        elif self.data == 'Kaggle':
            img_path = os.path.join(self.data_root, 'train_resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpeg')
        elif self.data == 'KaggleDR+':
            index = str(self.landmarks_frame.iloc[idx, 0])
            if os.path.exists(f'/mnt/data3_ssd/RetinalDataset/Kaggle/train_resized/{index}'):
                img_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/train_resized/', index)
            elif os.path.exists(f'/mnt/data3_ssd/RetinalDataset/Kaggle/valid_resized/{index}'):
                img_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/valid_resized/', index)
            else:
                print('Cannot find img path (KaggleDR+)')
                exit()
=======
            img_path = os.path.join(self.data_root, 'train/', self.landmarks_frame.iloc[idx, 0] + '.jpg')
        elif self.data == 'RFMiD':
            img_path = os.path.join(self.data_root, 'train/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        elif self.data == 'TAOP':
            img_path = os.path.join(self.data_root, 'png/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        elif self.data == 'APTOS':
            img_path = os.path.join(self.data_root, 'train_images/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        elif self.data == 'Kaggle':
            img_path = os.path.join(self.data_root, 'train/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpeg')
>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
        else:
            terminal_msg("Args.Data Error (From TrainDataset.__getitem__)", "F")
            exit()

        img = self.load_image(img_path)
        landmarks = np.array(self.landmarks_frame.iloc[idx, 1:], dtype=np.float32).tolist()

<<<<<<< HEAD
        if self.data in ["ODIR-5K", "RFMiD", "KaggleDR+"]:
=======
        if self.data in ["ODIR-5K", "RFMiD"]:
>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
            sample = {'image': img, 'landmarks': torch.tensor(landmarks).float()}
        elif self.data in ["TAOP", "APTOS", "Kaggle"]:
            sample = {'image': img, 'landmarks': torch.tensor(landmarks).int()}
        else:
            terminal_msg("Args.Data Error (From TrainDataset.__getitem__)", "F")
            exit()

        return sample

    def __len__(self):
<<<<<<< HEAD
       return len(self.landmarks_frame)
=======
        return len(self.landmarks_frame)

>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a

class ValidDataset(data.Dataset):
    """ 
    Based on the args.data to choose the dataset for validation:

    ODIR-5K: 693 samples
    RFMiD: 640 samples
<<<<<<< HEAD
    Kaggle: 5,721 samples

    TAOP: 297 samples
    APTOS: 367 samples
    Kaggle: 2,000 samples (for valid only)
=======

    TAOP: 297 samples
    APTOS: 367 samples
    Kaggle: 3513 samples
>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
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
<<<<<<< HEAD
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_test.csv')
        elif self.data == 'KaggleDR+':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/KaggleDR+/'
=======
>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_valid.csv')
        else:
            terminal_msg("Args.Data Error (From ValidDataset.__init__)", "F")
            exit()
<<<<<<< HEAD
                
=======

>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
    def load_image(self, path):
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.data == 'ODIR-5K':
<<<<<<< HEAD
            img_path = os.path.join(self.data_root, 'valid_resized/', self.landmarks_frame.iloc[idx, 0] + '.jpg')
        elif self.data == 'RFMiD':
            img_path = os.path.join(self.data_root, 'valid_resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        elif self.data == 'TAOP':
            img_path = os.path.join(self.data_root, 'png_resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        elif self.data == 'APTOS':
            img_path = os.path.join(self.data_root, 'train_resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        elif self.data == 'Kaggle':
            img_path = os.path.join(self.data_root, 'valid_resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpeg')
        elif self.data == 'KaggleDR+':
            index = str(self.landmarks_frame.iloc[idx, 0])
            if os.path.exists(f'/mnt/data3_ssd/RetinalDataset/Kaggle/train_resized/{index}'):
                img_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/train_resized/', index)
            elif os.path.exists(f'/mnt/data3_ssd/RetinalDataset/Kaggle/valid_resized/{index}'):
                img_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/valid_resized/', index)
            else:
                print('Cannot find img path (KaggleDR+)')
                exit()
=======
            img_path = os.path.join(self.data_root, 'valid/', self.landmarks_frame.iloc[idx, 0] + '.jpg')
        elif self.data == 'RFMiD':
            img_path = os.path.join(self.data_root, 'valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        elif self.data == 'TAOP':
            img_path = os.path.join(self.data_root, 'png/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        elif self.data == 'APTOS':
            img_path = os.path.join(self.data_root, 'train_images/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        elif self.data == 'Kaggle':
            img_path = os.path.join(self.data_root, 'train/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpeg')
>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
        else:
            terminal_msg("Args.Data Error (From ValidDataset.__getitem__)", "F")
            exit()

        img = self.load_image(img_path)
        landmarks = np.array(self.landmarks_frame.iloc[idx, 1:], dtype=np.float32).tolist()
        sample = {'image': img, 'landmarks': torch.tensor(landmarks).float()}

        return sample

    def __len__(self):
<<<<<<< HEAD
       return len(self.landmarks_frame)
=======
        return len(self.landmarks_frame)
>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
