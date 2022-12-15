<<<<<<< HEAD
from random import shuffle
import torch
import wandb
from torchsummary import summary
import warnings
import random
=======
from ast import arg
from random import shuffle
import torch
import pandas as pd
import wandb
from torchsummary import summary
import warnings
>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a

from models.build import build_single_task_model, build_hard_param_share_model

from engine.train import Single_Task_Trainer, Multi_Task_Trainer, setup_seed
from engine.eval import Single_Task_Evaluation, Multi_Task_Evaluation

from utils.info import terminal_msg, epic_start, get_device
from utils.parser import ParserArgs
from utils.model import save_checkpoint, resume_checkpoint

<<<<<<< HEAD
from data.dataset import TrainDataset, ValidDataset
from data.dataset import get_train_transforms, get_valid_transforms, get_valid_dataloader, get_train_data
=======
from data.dataset import TrainDataset, ValidDataset, get_train_transforms, get_valid_transforms, get_train_dataloader, get_valid_dataloader
>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    epic_start("Multi-task Learning on fundus image classification")
    args = ParserArgs().get_args()
<<<<<<< HEAD
    setup_seed(122500)
    device = get_device()

=======
    # setup_seed(122500)
    device = get_device()
>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
    if args.multi_task:
        model = build_hard_param_share_model(args)
    else:
        model = build_single_task_model(args)
    torch.backends.cudnn.benchmark = True

    if args.use_wandb:
<<<<<<< HEAD
        wandb.init(project= args.project)
=======
        wandb.init(project=args.project)
>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a

    train_transfrom = get_train_transforms(256)
    valid_transform = get_valid_transforms(256)

    if args.multi_task:
        terminal_msg("Processing the datasets for multi-task model", "E")
<<<<<<< HEAD
        train_data = get_train_data(args, train_transfrom)
=======
        train_dataloaders = get_train_dataloader(args, train_transfrom)
>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
        valid_dataloaders = get_valid_dataloader(args, valid_transform)
    else:
        terminal_msg("Processing the dataset for single-task model", "E")
        train_dataset = TrainDataset(args.data, transform=train_transfrom)
        valid_dataset = ValidDataset(args.data, transform=valid_transform)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    terminal_msg('DataLoader ({}) is ready!'.format(args.data), 'C')

    if args.multi_task:
        if args.mode == 'train':
<<<<<<< HEAD
            Multi_Task_Trainer(args, model, device, train_data, valid_dataloaders)
=======
            Multi_Task_Trainer(args, model, device, train_dataloaders, valid_dataloaders)
>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
        elif args.mode == 'eval':
            Multi_Task_Evaluation(args, model, device, valid_dataloaders)

    else:
        if args.mode == 'train':
            Single_Task_Trainer(args, model, device, train_dataloader, valid_dataloader)
        elif args.mode == 'eval':
<<<<<<< HEAD
            Single_Task_Evaluation(args, model, device, valid_dataloader)
=======
            Single_Task_Evaluation(args, model, device, valid_dataloader)
>>>>>>> 2f4b83349a47023660c13023fcf673789a76e64a
