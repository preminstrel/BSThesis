import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd
from PIL import Image
import os
from utils.info import terminal_msg
import numpy as np

from models.unet import UNET
from models.loss import DiceLoss, IoU
from models.discriminator import Discriminator
from models.loss import AdversarialLoss

from data.dataset import DriveDataset
from torch.utils.data import ConcatDataset
from torchvision.transforms import transforms
from utils.image import RandomGaussianBlur, get_color_distortion

import wandb
class TrainDataset(torch.utils.data.Dataset):
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
        elif self.data == 'AMD':
            img_path = os.path.join(self.data_root, 'resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
        elif self.data == 'DDR':
            img_path = os.path.join(self.data_root, 'resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
        elif self.data == 'LAG':
            img_path = os.path.join(self.data_root, 'train/', str(self.landmarks_frame.iloc[idx, 0]))
        elif self.data == 'PALM':
            img_path = os.path.join(self.data_root, 'resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
        elif self.data == 'REFUGE':
            img_path = os.path.join(self.data_root, 'resized/', str(self.landmarks_frame.iloc[idx, 0]) + '.jpg')
        else:
            terminal_msg("Args.Data Error (From TrainDataset.__getitem__)", "F")
            exit()

        img = self.load_image(img_path)
        landmarks = np.array(self.landmarks_frame.iloc[idx, 1:], dtype=np.float32).tolist()
        #sample = {'image': img, 'landmarks': torch.tensor(landmarks).float()}
        sample = img

        return sample

    def __len__(self):
       return len(self.landmarks_frame)

    def get_labels(self):
        return self.landmarks_frame.iloc[:,1:]

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


bs = 10
# DRIVE Dataloader
train_dataset = DriveDataset()

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=bs,
    shuffle=True,
    num_workers=2
)

# Model Initialization and setting up hyperparameters
model = UNET()
unet_ckpt = "/home/hssun/thesis/archive/checkpoints/unet.pth"
model.load_state_dict(torch.load(unet_ckpt))

model_D = Discriminator(in_channels=1)

lr = 1e-5 # [512, 1e-6] [224, 1e-6], [224, 1e-5], [224, 1e-4] [512, 1e-4] #
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# discriminator/generator learning rate ratio
d2g_lr = 0.1
lamd_gen = 1
optimizer_D = torch.optim.Adam(model_D.parameters(), lr=lr*d2g_lr)

epochs = 1000

# WandB – Initialize a new run
wandb.init(
    project="fundus_segmentation_domain_adaptation",
    config = {
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": bs,
        "d2g_lr": d2g_lr,
        "lamd_gen": lamd_gen,
    }
)


loss_fn1 = DiceLoss().cuda()
loss_fn2 = IoU().cuda()
adversarial_loss = AdversarialLoss().cuda()

# Taget Domain Datasets
all_dataset = []
data = "TAOP, APTOS, DDR, AMD, LAG, PALM, REFUGE, ODIR-5K, RFMiD, DR+"
data_dict = data.split(", ") # ['ODIR-5K', 'TAOP', 'RFMiD']
transform = transforms.Compose([
        transforms.RandomResizedCrop(size=512, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Compose([
            get_color_distortion(),
            RandomGaussianBlur(),
            ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
for data in data_dict:
    train_dataset = TrainDataset(data, transform=transform)
    all_dataset.append(train_dataset)

target_dataset = ConcatDataset(all_dataset)
print("All Datasets length:", len(target_dataset))
target_dataloader = DataLoader(target_dataset, batch_size=bs, shuffle=True, num_workers=2)
target_loader_iter = target_dataloader.__iter__()

# Train
for epoch in range(epochs):

    model.train()
    model_D.train()

    if torch.cuda.is_available():
        model.cuda()
        model_D.cuda()

    epoch_loss1 = 0.0
    epoch_loss2 = 0.0
    epoch_loss_D = 0.0
    epoch_loss_G = 0.0

    #with tqdm(train_loader, unit="batch") as tepoch:

    for idx, sample in enumerate(train_loader):
    
        x = sample['image']
        y = sample['mask']
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        image_target = next(target_loader_iter).cuda(non_blocking=True)

        # ---------------
        #  Source Domain
        # ---------------

        optimizer.zero_grad()
        y_pred = model(x)

        score1,loss1 = loss_fn1(y_pred, y)
        score2,loss2 = loss_fn2(y_pred, y)

        loss1.backward(retain_graph = True)
        loss2.backward()

        optimizer.step()

        epoch_loss1 += loss1.item()
        epoch_loss2 += loss2.item()

        # ---------------
        #  Target Domain
        # ---------------
        # zero optimizers
        optimizer.zero_grad()
        optimizer_D.zero_grad()

        # process_outputs
        output_target_mask = model(image_target)
        gen_loss = 0
        dis_loss = 0
        
        # discriminator loss
        dis_input_real = y_pred.detach()
        dis_input_fake = output_target_mask.detach()
        dis_real = model_D(dis_input_real)
        dis_fake = model_D(dis_input_fake)
        dis_real_loss = adversarial_loss(dis_real, True)
        dis_fake_loss = adversarial_loss(dis_fake, False)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2
        epoch_loss_D += dis_loss.item()
        
        dis_loss.backward()
        optimizer_D.step()

        # generator adversarial loss
        gen_input_fake = output_target_mask
        gen_fake = model_D(gen_input_fake)
        gen_gan_loss = adversarial_loss(gen_fake, True) * lamd_gen
        gen_loss += gen_gan_loss
        epoch_loss_G += gen_loss.item()
        
        gen_loss.backward()
        optimizer.step()

    epoch_loss1 = epoch_loss1/len(train_loader)
    epoch_loss2 = epoch_loss2/len(train_loader)
    if epoch % 100 == 0:
        wandb.log({"source img": [wandb.Image(x), wandb.Image(y),wandb.Image(y_pred)]})
        wandb.log({"target img": [wandb.Image(image_target), wandb.Image(output_target_mask)]})
    wandb.log({"Train Dice Loss": epoch_loss1, 
                "Train IoU Loss": epoch_loss2,
                "Train Discriminator Loss": epoch_loss_D,
                "Train Generator Loss": epoch_loss_G})

    print("Train Dice Loss: {}, ".format(epoch_loss1),"Train IoU Loss: {}, ".format(epoch_loss2))
    print("Saving model...")
    torch.save(model.state_dict(), "/home/hssun/thesis/archive/checkpoints/seg.pth")
    print("Model saved!")