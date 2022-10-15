from ast import arg
import torch.nn as nn
import torch
from termcolor import colored
import datetime
import time
import sys
import wandb
import cv2
from sklearn.metrics import roc_auc_score
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
import argparse
import scipy

from utils.info import terminal_msg, epic_start, get_device
from utils.parser import ParserArgs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.mode = self.args.mode

        # define the model

        # define the loss function
        cos_loss = torch.nn.CosineSimilarity().cuda()

        self.add_module("cos_loss", cos_loss)

        # define the optimizer

        # resume training
        if self.args.resume:
            terminal_msg("Loading model...", "E")
            ckpt_path = os.path.join("checkpoints", args.resume)
            if os.path.isfile(ckpt_path):
                rd_ckpt = torch.load(ckpt_path)
                # model.load_state_dict(rd_ckpt['decoder'])
                terminal_msg("Loaded checkpoint '{}'".format(args.resume), "C")
            else:
                terminal_msg("No checkpoint found!", "F")
        else:
            terminal_msg("Initializing model randomly...", "E")

    def process(self, image):
        pass

    def forward(self, image):
        pass

    def backward(self, loss=None):
        if loss is not None:
            loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    epic_start("My bachelor thesis")
    a = torch.randn([1, 3, 2])
