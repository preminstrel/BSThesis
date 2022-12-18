import argparse
import warnings
import os


class ParserArgs(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.get_general_parser()

        # ablation exps
        # self.get_ablation_exp_args()
        # comparison exps
        # self.get_comparison_exps_args()

    def get_general_parser(self):
        # training settings
        self.parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
        self.parser.add_argument("--batches", type=int, default=200, help="number of batches in an epoch")
        self.parser.add_argument("--batch_size", type=int, default=128, help="batch size")
        self.parser.add_argument("--num_workers", type=int, default=16, help="num_workers")
        self.parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        self.parser.add_argument("--resume", type=str, default="", metavar="PATH", help="path to checkpoints")
        self.parser.add_argument("--save_freq", type=int, default=50, help="save frequency")
        self.parser.add_argument("--valid_freq", type=int, default=1, help="validation frequency")
        self.parser.add_argument("--multi_gpus", action="store_true", help="enable DataParallel")

        # eval settings
        self.parser.add_argument("--use_wandb", action="store_true", help="enable wandb")
        self.parser.add_argument("--project", type=str, help="project name for wandb")
        self.parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "viusalize"])

        # data settings
        self.parser.add_argument("--data", type=str, default="TAOP")
        self.parser.add_argument("--image_size", type=int, default=224, help="image size")
        self.parser.add_argument("--balanced_sampling", action="store_true", help="enable wandb")

        # multi-task learning
        self.parser.add_argument("--multi_task", action="store_true", help="enable multi-task")
        self.parser.add_argument("--method", type=str, default="HPS")

    def get_args(self):
        args = self.parser.parse_args()
        return args
