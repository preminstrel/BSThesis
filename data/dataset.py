import os
import numpy as np
from PIL import Image
import torch
import random
from torch.utils import data
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
import pandas as pd
from torch.utils.data import DataLoader
import glob

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

#===========================================Transforms===============================================#

def get_train_transforms(img_size):
    r'''
    resize (256)-> random crop (224) -> random filp -> color distortion -> GaussianBlur -> normalize
    '''
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
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
    r'''
    resize (224)-> normalize
    '''
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

#==============================================APIs==================================================#

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
    if "AMD" in args.data:
        datasets["AMD"] = TrainDataset('AMD', transform)
    if "DDR" in args.data:
        datasets["DDR"] = TrainDataset('DDR', transform)
    if "LAG" in args.data:
        datasets["LAG"] = TrainDataset('LAG', transform)
    if "PALM" in args.data:
        datasets["PALM"] = TrainDataset('PALM', transform)
    if "REFUGE" in args.data:
        datasets["REFUGE"] = TrainDataset('REFUGE', transform)
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
    if "AMD" in args.data:
        datasets["AMD"] = ValidDataset('AMD', transform)
    if "DDR" in args.data:
        datasets["DDR"] = ValidDataset('DDR', transform)
    if "LAG" in args.data:
        datasets["LAG"] = ValidDataset('LAG', transform)
    if "PALM" in args.data:
        datasets["PALM"] = ValidDataset('PALM', transform)
    if "REFUGE" in args.data:
        datasets["REFUGE"] = ValidDataset('REFUGE', transform)
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
    if "AMD" in args.data:
        dataloaders["AMD"] = DataLoader(TrainDataset('AMD', transform), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    if "DDR" in args.data:
        dataloaders["DDR"] = DataLoader(TrainDataset('DDR', transform), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    if "LAG" in args.data:
        dataloaders["LAG"] = DataLoader(TrainDataset('LAG', transform), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    if "PALM" in args.data:
        dataloaders["PALM"] = DataLoader(TrainDataset('PALM', transform), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    if "REFUGE" in args.data:
        dataloaders["REFUGE"] = DataLoader(TrainDataset('REFUGE', transform), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    else:
        terminal_msg("Args.Data Error (From get_train_dataloder)", "F")
        exit()
    return dataloaders

def get_valid_dataloader(args, transform):
    dataloaders = {}
    if "ODIR-5K" in args.data:
        dataloaders["ODIR-5K"] = DataLoader(ValidDataset('ODIR-5K', transform, args=args), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    if "RFMiD" in args.data:
        dataloaders["RFMiD"] = DataLoader(ValidDataset('RFMiD', transform, args=args), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    if "TAOP" in args.data:
        dataloaders["TAOP"] = DataLoader(ValidDataset('TAOP', transform, args=args), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    if "APTOS" in args.data:
        dataloaders["APTOS"] = DataLoader(ValidDataset('APTOS', transform, args=args), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    if "Kaggle" in args.data:
        dataloaders["Kaggle"] = DataLoader(ValidDataset('Kaggle', transform, args=args), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    if "DR+" in args.data:
        dataloaders["DR+"] = DataLoader(ValidDataset('DR+', transform, args=args), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    if "AMD" in args.data:
        dataloaders["AMD"] = DataLoader(ValidDataset('AMD', transform, args=args), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    if "DDR" in args.data:
        dataloaders["DDR"] = DataLoader(ValidDataset('DDR', transform, args=args), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    if "LAG" in args.data:
        dataloaders["LAG"] = DataLoader(ValidDataset('LAG', transform, args=args), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    if "PALM" in args.data:
        dataloaders["PALM"] = DataLoader(ValidDataset('PALM', transform, args=args), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    if "REFUGE" in args.data:
        dataloaders["REFUGE"] = DataLoader(ValidDataset('REFUGE', transform, args=args), batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
    assert dataloaders
    return dataloaders

def get_train_data(args, transform):
    data = {}

    if 'ODIR-5K' in args.data:
        data['ODIR-5K'] = {}
        train_dataset = TrainDataset('ODIR-5K', transform, args=args)
        if args.balanced_sampling:
            data['ODIR-5K']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, sampler=MultilabelBalancedRandomSampler(train_dataset.get_labels()), pin_memory= True, num_workers=args.num_workers)
        else:
            data['ODIR-5K']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
        data['ODIR-5K']['iterloader'] = iter(data['ODIR-5K']['dataloader'])
    
    if 'RFMiD' in args.data:
        data['RFMiD'] = {}
        train_dataset = TrainDataset('RFMiD', transform, args=args)
        if args.balanced_sampling:
            data['RFMiD']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, sampler=MultilabelBalancedRandomSampler(train_dataset.get_labels()), pin_memory= True, num_workers=args.num_workers)
        else:
            data['RFMiD']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
        data['RFMiD']['iterloader'] = iter(data['RFMiD']['dataloader'])
    
    if 'DR+' in args.data:
        data['DR+'] = {}
        data['DR+'] = {}
        train_dataset = TrainDataset('DR+', transform, args=args)
        if args.balanced_sampling:
            data['DR+']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, sampler=MultilabelBalancedRandomSampler(train_dataset.get_labels()), pin_memory= True, num_workers=args.num_workers)
        else:
            data['DR+']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
        data['DR+']['iterloader'] = iter(data['DR+']['dataloader'])
    
    if 'TAOP' in args.data:
        data['TAOP'] = {}
        train_dataset = TrainDataset('TAOP', transform, args=args)
        if args.balanced_sampling:
            data['TAOP']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, sampler=ImbalancedDatasetSampler(train_dataset), pin_memory= True, num_workers=args.num_workers)
        else:
            data['TAOP']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
        data['TAOP']['iterloader'] = iter(data['TAOP']['dataloader'])
    
    if 'APTOS' in args.data:
        data['APTOS'] = {}
        train_dataset = TrainDataset('APTOS', transform, args=args)
        if args.balanced_sampling:
            data['APTOS']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, sampler=ImbalancedDatasetSampler(train_dataset), pin_memory= True, num_workers=args.num_workers)
        else:
            data['APTOS']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
        data['APTOS']['iterloader'] = iter(data['APTOS']['dataloader'])
    
    if 'Kaggle' in args.data:
        data['Kaggle'] = {}
        train_dataset = TrainDataset('Kaggle', transform, args=args)
        if args.balanced_sampling:
            data['Kaggle']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, sampler=ImbalancedDatasetSampler(train_dataset), pin_memory= True, num_workers=args.num_workers)
        else:
            data['Kaggle']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
        data['Kaggle']['iterloader'] = iter(data['Kaggle']['dataloader'])
    
    if 'AMD' in args.data:
        data['AMD'] = {}
        train_dataset = TrainDataset('AMD', transform, args=args)
        if args.balanced_sampling:
            data['AMD']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, sampler=ImbalancedDatasetSampler(train_dataset), pin_memory= True, num_workers=args.num_workers)
        else:
            data['AMD']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
        data['AMD']['iterloader'] = iter(data['AMD']['dataloader'])
    
    if 'DDR' in args.data:
        data['DDR'] = {}
        train_dataset = TrainDataset('DDR', transform, args=args)
        if args.balanced_sampling:
            data['DDR']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, sampler=ImbalancedDatasetSampler(train_dataset), pin_memory= True, num_workers=args.num_workers)
        else:
            data['DDR']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
        data['DDR']['iterloader'] = iter(data['DDR']['dataloader'])
    
    if 'LAG' in args.data:
        data['LAG'] = {}
        train_dataset = TrainDataset('LAG', transform, args=args)
        if args.balanced_sampling:
            data['LAG']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, sampler=ImbalancedDatasetSampler(train_dataset), pin_memory= True, num_workers=args.num_workers)
        else:
            data['LAG']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
        data['LAG']['iterloader'] = iter(data['LAG']['dataloader'])
    
    if 'RFMiD' in args.data:
        data['RFMiD'] = {}
        train_dataset = TrainDataset('RFMiD', transform, args=args)
        if args.balanced_sampling:
            data['RFMiD']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, sampler=ImbalancedDatasetSampler(train_dataset), pin_memory= True, num_workers=args.num_workers)
        else:
            data['RFMiD']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
        data['RFMiD']['iterloader'] = iter(data['RFMiD']['dataloader'])
    
    if 'PALM' in args.data:
        data['PALM'] = {}
        train_dataset = TrainDataset('PALM', transform, args=args)
        if args.balanced_sampling:
            data['PALM']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, sampler=ImbalancedDatasetSampler(train_dataset), pin_memory= True, num_workers=args.num_workers)
        else:
            data['PALM']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
        data['PALM']['iterloader'] = iter(data['PALM']['dataloader'])
    
    if 'REFUGE' in args.data:
        data['REFUGE'] = {}
        train_dataset = TrainDataset('REFUGE', transform, args=args)
        if args.balanced_sampling:
            data['REFUGE']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, sampler=ImbalancedDatasetSampler(train_dataset), pin_memory= True, num_workers=args.num_workers)
        else:
            data['REFUGE']['dataloader'] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory= True, num_workers=args.num_workers)
        data['REFUGE']['iterloader'] = iter(data['REFUGE']['dataloader'])
    
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
    train_dataset = TrainDataset(data=args.data, transform=train_transfrom, args=args)
    valid_dataset = ValidDataset(data=args.data, transform=valid_transform, args=args)

    if args.balanced_sampling:
        if args.data in ["ODIR-5K", "RFMiD", "DR+"]:
            sampler = MultilabelBalancedRandomSampler(train_dataset.get_labels())
        elif args.data in ["TAOP", "APTOS", "Kaggle", "AMD", "DDR", "LAG", "PALM"]:
            sampler = ImbalancedDatasetSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    return train_dataloader, valid_dataloader

#============================================Datasets================================================#

class TrainDataset(data.Dataset):
    r""" 
    Based on the args.data to choose the dataset for training:

    ODIR-5K: 6,307 samples
    RFMiD: 1,920 samples
    DR+: 51,491 samples

    TAOP: 3,000 samples
    APTOS: 3,295 samples
    Kaggle: 35,126 samples
    DDR: 6,835 samples

    AMD: 321 samples
    LAG: 3,884 samples
    PALM: 641 samples
    REFUGE: 400 samples
    """

    def __init__(self, data, transform=None, args=None):
        self.transform = transform
        self.data = data
        self.args = args

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
        elif self.data == 'AMD':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/iChallenge-AMD/Training400/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_train.csv')
        elif self.data == 'DDR':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/DDR/DDR-dataset/DR_grading/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_train.csv')
        elif self.data == 'LAG':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/LAG/dataset/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_train.csv')
        elif self.data == 'PALM':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/iChallenge-PM/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_train.csv')
        elif self.data == 'REFUGE':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/REFUGE/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_train.csv')
        elif self.data == 'IDRiD':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/IDRiD/B. Disease Grading/'
            self.landmarks_frame = pd.read_csv(self.data_root + '2. Groundtruths/train_grade.csv')
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
            if self.args.preprocessed:
                img_path = os.path.join(self.data_root, 'train_CLAHE/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
                ma_path = os.path.join(self.data_root, 'ma_train/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
                od_path = os.path.join(self.data_root, 'od_train/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
            else:
                img_path = os.path.join(self.data_root, 'train_resized/', self.landmarks_frame.iloc[idx, 0] + '.jpg')
                ma_path = os.path.join(self.data_root, 'ma_train/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
                od_path = os.path.join(self.data_root, 'od_train/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
        elif self.data == 'RFMiD':
            if self.args.preprocessed:
                img_path = os.path.join(self.data_root, 'train_CLAHE/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
                ma_path = os.path.join(self.data_root, 'ma_train/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
                od_path = os.path.join(self.data_root, 'od_train/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
            else:
                img_path = os.path.join(self.data_root, 'train_resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
                ma_path = os.path.join(self.data_root, 'ma_train/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
                od_path = os.path.join(self.data_root, 'od_train/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        elif self.data == 'TAOP':
            if self.args.preprocessed:
                img_path = os.path.join(self.data_root, 'CLAHE/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
                ma_path = os.path.join(self.data_root, 'ma_train/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
                od_path = os.path.join(self.data_root, 'od_train/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
            else:
                img_path = os.path.join(self.data_root, 'png_resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
                ma_path = os.path.join(self.data_root, 'ma_train/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
                od_path = os.path.join(self.data_root, 'od_train/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        elif self.data == 'APTOS':
            if self.args.preprocessed:
                img_path = os.path.join(self.data_root, 'CLAHE/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
                ma_path = os.path.join(self.data_root, 'ma_train/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
                od_path = os.path.join(self.data_root, 'od_train/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
            else:
                img_path = os.path.join(self.data_root, 'train_resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
                ma_path = os.path.join(self.data_root, 'ma_train/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
                od_path = os.path.join(self.data_root, 'od_train/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        elif self.data == 'Kaggle':
            img_path = os.path.join(self.data_root, 'train_resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpeg')
            ma_path = os.path.join(self.data_root, 'ma_train/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpeg')
            od_path = os.path.join(self.data_root, 'od_train/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpeg')
        elif self.data == 'DR+':
            if self.args.preprocessed:
                img_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/CLAHE/', str(self.landmarks_frame.iloc[idx, 0]))
                ma_path = os.path.join(self.data_root, 'ma_train/', str(self.landmarks_frame.iloc[idx, 0]))
                od_path = os.path.join(self.data_root, 'od_train/', str(self.landmarks_frame.iloc[idx, 0]))
            else:
                index = str(self.landmarks_frame.iloc[idx, 0])
                if os.path.exists(f'/mnt/data3_ssd/RetinalDataset/Kaggle/train_resized/{index}'):
                    img_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/train_resized/', index)
                    # ma_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/ma_train/', index)
                    # od_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/od_train/', index)
                elif os.path.exists(f'/mnt/data3_ssd/RetinalDataset/Kaggle/valid_resized/{index}'):
                    img_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/valid_resized/', index)
                    #ma_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/ma_valid/', index)
                    #od_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/od_valid/', index)
                else:
                    print('Cannot find img path (DR+)')
                if os.path.exists(f'/mnt/data3_ssd/RetinalDataset/Kaggle/ma_train/{index}'):
                    ma_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/ma_train/', index)
                elif os.path.exists(f'/mnt/data3_ssd/RetinalDataset/Kaggle/ma_valid/{index}'):
                    ma_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/ma_valid/', index)
                else:
                    print('Cannot find ma path (DR+)')
                if os.path.exists(f'/mnt/data3_ssd/RetinalDataset/Kaggle/od_train/{index}'):
                    od_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/od_train/', index)
                elif os.path.exists(f'/mnt/data3_ssd/RetinalDataset/Kaggle/ma_valid/{index}'):
                    od_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/od_valid/', index)
                else:
                    print('Cannot find ma path (DR+)')
        
        elif self.data == 'AMD':
            if self.args.preprocessed:
                img_path = os.path.join(self.data_root, 'CLAHE/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
            else:
                img_path = os.path.join(self.data_root, 'resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
                ma_path = os.path.join('/mnt/data3_ssd/RetinalDataset/iChallenge-AMD/', 'ma_train/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
                od_path = os.path.join('/mnt/data3_ssd/RetinalDataset/iChallenge-AMD/', 'od_train/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
        elif self.data == 'DDR':
            if self.args.preprocessed:
                img_path = os.path.join(self.data_root, 'CLAHE/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
            else:
                img_path = os.path.join(self.data_root, 'resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
                ma_path = os.path.join(self.data_root, 'ma_train/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
                od_path = os.path.join(self.data_root, 'od_train/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
        elif self.data == 'LAG':
            if self.args.preprocessed:
                img_path = os.path.join(self.data_root, 'train_CLAHE/', str(self.landmarks_frame.iloc[idx, 0]))
            else:
                img_path = os.path.join(self.data_root, 'train/', str(self.landmarks_frame.iloc[idx, 0]))
                ma_path = os.path.join(self.data_root, 'ma_train/', str(self.landmarks_frame.iloc[idx, 0]))
                od_path = os.path.join(self.data_root, 'od_train/', str(self.landmarks_frame.iloc[idx, 0]))
        elif self.data == 'PALM':
            if self.args.preprocessed:
                img_path = os.path.join(self.data_root, 'CLAHE/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
            else:
                img_path = os.path.join(self.data_root, 'resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
                ma_path = os.path.join(self.data_root, 'ma_train/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
                od_path = os.path.join(self.data_root, 'od_train/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
        elif self.data == 'REFUGE':
            if self.args.preprocessed:
                img_path = os.path.join(self.data_root, 'CLAHE/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
                ma_path = os.path.join(self.data_root, 'ma_train/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
                od_path = os.path.join(self.data_root, 'od_train/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
            else:
                img_path = os.path.join(self.data_root, 'resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
                ma_path = os.path.join(self.data_root, 'ma_train/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
                od_path = os.path.join(self.data_root, 'od_train/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
        elif self.data == 'IDRiD':
            img_path = os.path.join(self.data_root, '1. Original Images/a. Training Set/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
        else:
            terminal_msg("Args.Data Error (From TrainDataset.__getitem__)", "F")
            exit()

        img = self.load_image(img_path)
        # ma_img = self.load_image(ma_path)
        # od_img = self.load_image(od_path)
        landmarks = np.array(self.landmarks_frame.iloc[idx, 1:], dtype=np.float32).tolist()
        sample = {'image': img, 'landmarks': torch.tensor(landmarks).float()}

        return sample

    def __len__(self):
       return len(self.landmarks_frame)

    def get_labels(self):
        return self.landmarks_frame.iloc[:,1:]

class ValidDataset(data.Dataset):
    r""" 
    Based on the args.data to choose the dataset for validation:

    ODIR-5K: 693 samples
    RFMiD: 640 samples
    DR+: 5,722 samples

    TAOP: 297 samples
    APTOS: 367 samples
    Kaggle: 2,000 samples (for valid only)
    DDR: 2,733 samples
    
    AMD: 79 samples
    LAG: 970 samples
    PALM: 159 samples
    REFUGE: 400 samples
    """

    def __init__(self, data, transform=None, args=None):
        self.transform = transform
        self.data = data
        self.args = args

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
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_valid.csv')
        elif self.data == 'DR+':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/KaggleDR+/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_valid.csv')
        elif self.data == 'AMD':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/iChallenge-AMD/Training400/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_valid.csv')
        elif self.data == 'DDR':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/DDR/DDR-dataset/DR_grading/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_valid.csv')
        elif self.data == 'LAG':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/LAG/dataset/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_valid.csv')
        elif self.data == 'PALM':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/iChallenge-PM/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_valid.csv')
        elif self.data == 'REFUGE':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/REFUGE/'
            self.landmarks_frame = pd.read_csv(self.data_root + 'label_valid.csv')
        elif self.data == 'IDRiD':
            self.data_root = '/mnt/data3_ssd/RetinalDataset/IDRiD/B. Disease Grading/'
            self.landmarks_frame = pd.read_csv(self.data_root + '2. Groundtruths/test_grade.csv')
        else:
            terminal_msg("Args.Data Error (From ValidDataset.__init__)", "F")
                
    def load_image(self, path):
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.data == 'ODIR-5K':
            if self.args.preprocessed:
                img_path = os.path.join(self.data_root, 'valid_CLAHE/', self.landmarks_frame.iloc[idx, 0] + '.jpg')
            else:
                img_path = os.path.join(self.data_root, 'valid_resized/', self.landmarks_frame.iloc[idx, 0] + '.jpg')
                ma_path = os.path.join(self.data_root, 'ma_valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
                od_path = os.path.join(self.data_root, 'od_valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
        elif self.data == 'RFMiD':
            if self.args.preprocessed:
                img_path = os.path.join(self.data_root, 'valid_CLAHE/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
                ma_path = os.path.join(self.data_root, 'ma_valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
                od_path = os.path.join(self.data_root, 'od_valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
            else:
                img_path = os.path.join(self.data_root, 'valid_resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
                ma_path = os.path.join(self.data_root, 'ma_valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
                od_path = os.path.join(self.data_root, 'od_valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        elif self.data == 'TAOP':
            if self.args.preprocessed:
                img_path = os.path.join(self.data_root, 'CLAHE/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
                ma_path = os.path.join(self.data_root, 'ma_valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
                od_path = os.path.join(self.data_root, 'od_valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
            else:
                img_path = os.path.join(self.data_root, 'png_resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
                ma_path = os.path.join(self.data_root, 'ma_valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
                od_path = os.path.join(self.data_root, 'od_valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        elif self.data == 'APTOS':
            if self.args.preprocessed:
                img_path = os.path.join(self.data_root, 'CLAHE/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
            else:
                img_path = os.path.join(self.data_root, 'train_resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
                ma_path = os.path.join(self.data_root, 'ma_valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
                od_path = os.path.join(self.data_root, 'od_valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.png')
        elif self.data == 'Kaggle':
            img_path = os.path.join(self.data_root, 'valid_resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpeg')
            ma_path = os.path.join(self.data_root, 'ma_valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
            od_path = os.path.join(self.data_root, 'od_valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
        elif self.data == 'DR+':
            if self.args.preprocessed:
                img_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/CLAHE/', str(self.landmarks_frame.iloc[idx, 0]))
            else:
                index = str(self.landmarks_frame.iloc[idx, 0])
                if os.path.exists(f'/mnt/data3_ssd/RetinalDataset/Kaggle/train_resized/{index}'):
                    img_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/train_resized/', index)
                    # ma_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/ma_train/', index)
                    # od_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/od_train/', index)
                elif os.path.exists(f'/mnt/data3_ssd/RetinalDataset/Kaggle/valid_resized/{index}'):
                    img_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/valid_resized/', index)
                    #ma_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/ma_valid/', index)
                    #od_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/od_valid/', index)
                else:
                    print('Cannot find img path (DR+)')
                if os.path.exists(f'/mnt/data3_ssd/RetinalDataset/Kaggle/ma_train/{index}'):
                    ma_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/ma_train/', index)
                elif os.path.exists(f'/mnt/data3_ssd/RetinalDataset/Kaggle/ma_valid/{index}'):
                    ma_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/ma_valid/', index)
                else:
                    print('Cannot find ma path (DR+)')
                if os.path.exists(f'/mnt/data3_ssd/RetinalDataset/Kaggle/od_train/{index}'):
                    od_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/od_train/', index)
                elif os.path.exists(f'/mnt/data3_ssd/RetinalDataset/Kaggle/ma_valid/{index}'):
                    od_path = os.path.join('/mnt/data3_ssd/RetinalDataset/Kaggle/od_valid/', index)
                else:
                    print('Cannot find ma path (DR+)')
        elif self.data == 'AMD':
            if self.args.preprocessed:
                img_path = os.path.join(self.data_root, 'CLAHE/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
            else:
                img_path = os.path.join(self.data_root, 'resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
                ma_path = os.path.join('/mnt/data3_ssd/RetinalDataset/iChallenge-AMD/', 'ma_valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
                od_path = os.path.join('/mnt/data3_ssd/RetinalDataset/iChallenge-AMD/', 'od_valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
        elif self.data == 'DDR':
            if self.args.preprocessed:
                img_path = os.path.join(self.data_root, 'CLAHE/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
                ma_path = os.path.join(self.data_root, 'ma_valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
                od_path = os.path.join(self.data_root, 'od_valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
            else:
                img_path = os.path.join(self.data_root, 'resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
                ma_path = os.path.join(self.data_root, 'ma_valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
                od_path = os.path.join(self.data_root, 'od_valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
        elif self.data == 'LAG':
            if self.args.preprocessed:
                img_path = os.path.join(self.data_root, 'valid_CLAHE/', str(self.landmarks_frame.iloc[idx, 0]))
                ma_path = os.path.join(self.data_root, 'ma_valid/', str(self.landmarks_frame.iloc[idx, 0]))
                od_path = os.path.join(self.data_root, 'od_valid/', str(self.landmarks_frame.iloc[idx, 0]))
            else:
                img_path = os.path.join(self.data_root, 'valid/', str(self.landmarks_frame.iloc[idx, 0]))
                ma_path = os.path.join(self.data_root, 'ma_valid/', str(self.landmarks_frame.iloc[idx, 0]) )
                od_path = os.path.join(self.data_root, 'od_valid/', str(self.landmarks_frame.iloc[idx, 0]) )
        elif self.data == 'PALM':
            if self.args.preprocessed:
                img_path = os.path.join(self.data_root, 'CLAHE/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
            else:
                img_path = os.path.join(self.data_root, 'resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
                ma_path = os.path.join(self.data_root, 'ma_valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
                od_path = os.path.join(self.data_root, 'od_valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
        elif self.data == 'REFUGE':
            if self.args.preprocessed:
                img_path = os.path.join(self.data_root, 'CLAHE/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
                ma_path = os.path.join(self.data_root, 'ma_valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
                od_path = os.path.join(self.data_root, 'od_valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
            else:
                img_path = os.path.join(self.data_root, 'resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
                ma_path = os.path.join(self.data_root, 'ma_valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
                od_path = os.path.join(self.data_root, 'od_valid/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
        elif self.data == 'IDRiD':
            img_path = os.path.join(self.data_root, '1. Original Images/b. Testing Set/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
        else:
            terminal_msg("Args.Data Error (From ValidDataset.__getitem__)", "F")
            exit()

        img = self.load_image(img_path)
        # ma_img = self.load_image(ma_path)
        # od_img = self.load_image(od_path)
        landmarks = np.array(self.landmarks_frame.iloc[idx, 1:], dtype=np.float32).tolist()
        sample = {'image': img, 'landmarks': torch.tensor(landmarks).float()}

        return sample

    def __len__(self):
       return len(self.landmarks_frame)

def merge_datasets(dataset, sub_dataset):
    # samples
    dataset.samples.extend(sub_dataset.samples)

class DriveDataset(data.Dataset):
    def __init__(self):
        self.images_path = glob.glob('/home/hssun/seg_dataset/training/images/*.tif')
        self.masks_path = glob.glob('/home/hssun/seg_dataset/training/1st_manual/*.gif')
        self.n_samples = len(self.images_path)
        self.transform = ComposeWithMask()

    def __getitem__(self, index):
        """ Reading image """
        img = self.load_image(self.images_path[index])
        mask_file = self.images_path[index].split('/')[-1].replace('training', 'manual1').replace('tif', 'gif')
        mask_path = os.path.join('/home/hssun/seg_dataset/training/1st_manual/', mask_file)
        mask = self.load_image(mask_path, mode='L')
        img, mask = self.transform(img, mask)
        sample = {'image': img, 'mask': mask}

        return sample
    
    def load_image(self, path, mode='RGB'):
        image = Image.open(path).convert(mode)
        return image

    def __len__(self):
        return self.n_samples
    
class ComposeWithMask(object):
    def __init__(self):
        self.transforms = transforms.Compose([
            get_color_distortion(),
            RandomGaussianBlur(),
            ]),
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __call__(self, img, mask):
        # 随机裁剪
        i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333))
        img = TF.resize(TF.crop(img, i, j, h, w), (512,512))
        mask = TF.resize(TF.crop(mask, i, j, h, w), (512,512))
        # img = TF.resize(img, (224,224))
        # mask = TF.resize(mask, (224,224))

        # 随机翻转
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        
        #随机旋转
        degree = transforms.RandomRotation.get_params([-90, 90])
        img = TF.rotate(img, degree)
        mask = TF.rotate(mask, degree)

        # 应用其他变换
        for t in self.transforms:
            img = t(img)
        
        img = transforms.ToTensor()(img)
        mask = transforms.ToTensor()(mask)
        
        img = self.normalize(img)

        return img, mask