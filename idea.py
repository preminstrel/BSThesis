from random import shuffle
import torch
import pandas as pd
import wandb
from torchsummary import summary

from models.build import build_single_task_model

from engine.train import Trainer, setup_seed
from engine.eval import Evaluation

from utils.info import terminal_msg, epic_start, get_device
from utils.parser import ParserArgs

from data.dataset import TrainDataset, ValidDataset, get_transforms


if __name__ == "__main__":
    epic_start("Multi-task Learning on fundus image classification")
    args = ParserArgs().get_args()
    setup_seed(122500)
    model = build_single_task_model(args)
    device = get_device()
    model.to(device)
    torch.backends.cudnn.benchmark = True
    if args.use_wandb:
        wandb.init(project=args.project)

    terminal_msg("Processing the dataset", "E")
    transfrom = get_transforms(256)
    train_dataset = TrainDataset(args, transform=transfrom)
    valid_dataset = ValidDataset(args, transform=transfrom)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    terminal_msg('DataLoader ({}) is ready!'.format(args.data), 'C')

    if args.mode == 'train':
        Trainer(args, model, device, train_dataloader, valid_dataloader)
    elif args.mode == 'eval':
        Evaluation(args, model, device, valid_dataloader)
