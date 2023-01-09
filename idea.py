from random import shuffle
import torch
import wandb
from torchsummary import summary
import warnings
import os

from models.build import build_single_task_model, build_HPS_model, build_MMoE_model, build_CGC_model, build_MTAN_model, build_DSelectK_model, build_LTB_model

from engine.train import Single_Task_Trainer, Multi_Task_Trainer, setup_seed
from engine.eval import Single_Task_Evaluation, Multi_Task_Evaluation

from utils.info import terminal_msg, epic_start, get_device
from utils.parser import ParserArgs
from utils.model import save_checkpoint, resume_checkpoint

from data.dataset import get_train_transforms, get_valid_transforms, get_valid_dataloader, get_train_data, get_single_task_train_dataloader


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    epic_start("Multi-task Learning on fundus image classification")
    args = ParserArgs().get_args()
    setup_seed(122500)
    device = get_device()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    if args.multi_task:
        if args.method == "HPS":
            model = build_HPS_model(args)
        elif args.method == "MMoE":
            model = build_MMoE_model(args)
            #print(model)
        elif args.method == "CGC":
            model = build_CGC_model(args)
        elif args.method == "MTAN":
            model = build_MTAN_model(args)
        elif args.method == "DSelectK":
            model = build_DSelectK_model(args)
        elif args.method == "LTB":
            model = build_LTB_model(args)
        else:
            terminal_msg(f"Wrong mothod {args.method}", "F")
    else:
        model = build_single_task_model(args)
    torch.backends.cudnn.benchmark = True
    
    if args.multi_gpus:
        model = torch.nn.DataParallel(model)

    if args.use_wandb:
        wandb.init(project= args.project)
        wandb.watch(model)

    train_transfrom = get_train_transforms(224)
    valid_transform = get_valid_transforms(224)

    if args.multi_task:
        terminal_msg("Processing the datasets for multi-task model", "E")
        train_data = get_train_data(args, train_transfrom)
        valid_dataloaders = get_valid_dataloader(args, valid_transform)
    else:
        terminal_msg("Processing the dataset for single-task model", "E")
        train_dataloader, valid_dataloader = get_single_task_train_dataloader(args, train_transfrom, valid_transform)
    terminal_msg('DataLoader ({}) is ready!'.format(args.data), 'C')

    if args.multi_task:
        if args.mode == 'train':
            Multi_Task_Trainer(args, model, device, train_data, valid_dataloaders)
        elif args.mode == 'eval':
            Multi_Task_Evaluation(args, model, device, valid_dataloaders)

    else:
        if args.mode == 'train':
            Single_Task_Trainer(args, model, device, train_dataloader, valid_dataloader)
        elif args.mode == 'eval':
            Single_Task_Evaluation(args, model, device, valid_dataloader)
