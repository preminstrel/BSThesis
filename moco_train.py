#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import ConcatDataset
from torch.utils import data
import os
import numpy as np
from PIL import Image
import torch
import random
from torch.utils import data
from torchvision.transforms import transforms
import pandas as pd
from torch.utils.data import DataLoader

from data.dataset import ValidDataset
from data.dataset import get_data_weights, get_batch
from utils.image import GaussianBlur, TwoCropsTransform
from models.moco import MoCo


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=list,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--data', default="TAOP, APTOS, DDR, AMD, LAG, PALM, REFUGE, ODIR-5K, RFMiD, DR+",)

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')


def main():
    print('Jump into main...')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count() #2
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    print('Jump into main_worker...')
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if True:
        print('debug')
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if True:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + 0
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = MoCo(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    #print(model)

        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=args.gpu)
    else:
        model.cuda()
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    all_dataset = []
    data = "TAOP, APTOS, DDR, AMD, LAG, PALM, REFUGE, ODIR-5K, RFMiD, DR+"
    data_dict = data.split(", ") # ['ODIR-5K', 'TAOP', 'RFMiD']
    for data in data_dict:
        train_dataset = TrainDataset(data, transform = TwoCropsTransform(transforms.Compose(augmentation)))
        all_dataset.append(train_dataset)
        print(data, len(train_dataset))

    train_dataset = ConcatDataset(all_dataset)
    print("All Datasets:", len(ConcatDataset(all_dataset)))


    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=0, pin_memory=True, sampler=train_sampler, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            if (epoch + 1) % 10 == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=False, filename='archive/checkpoints/pretrained/pretrained_resnet50_{:04d}.pth'.format(epoch))


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    for i, sample in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = sample

        images[0] = images[0].cuda(args.gpu, non_blocking=True)
        images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        output, target = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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

if __name__ == '__main__':
    main()