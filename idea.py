import torch.nn as nn
import torch
import pandas as pd
import wandb
from torchsummary import summary

from models.build import build_model
from models.encoder_decoder import Decoder, Encoder, BasicBlock, Bottleneck

from engine.train import Trainer, setup_seed

from utils.info import terminal_msg, epic_start, get_device
from utils.parser import ParserArgs

from data.dataset import TrainDataset, get_transforms


if __name__ == "__main__":
    epic_start("Multi-task Learning on fundus image classification")
    args = ParserArgs().get_args()
    setup_seed(123)
    model = build_model(args)
    device = get_device()
    model.to(device)

    if args.use_wandb:
        wandb.init(project=args.project)

    terminal_msg("Processing the dataset", "E")
    transfrom = get_transforms(256)
    train_dataset = TrainDataset(args, transform=transfrom)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    terminal_msg('DataLoader ({}) is ready!'.format(args.data), 'C')

    Trainer(args, model, device, train_dataloader)
    # for i, sample in enumerate(dataloader):
    #     img = sample['image']
    #     label = sample['landmarks']
    #summary(model, (3, 256, 256))
