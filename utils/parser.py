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
        self.parser.add_argument("--batch_size", type=int, default=16, help="batch size")
        self.parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
        self.parser.add_argument("--resume", type=str, default="", metavar="PATH", help="path to checkpoints")
        self.parser.add_argument("--save_freq", type=int, default=10, help="save frequency")
        self.parser.add_argument("--valid_freq", type=int, default=10, help="validation frequency")
        self.parser.add_argument("--device_ids", type=list, default=[0], help="device ids")

        # eval settings
        self.parser.add_argument("--use_wandb", action="store_true", help="enable wandb")
        self.parser.add_argument("--project", type=str, help="project name for wandb")
        self.parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "viusalize"])

        # data settings
        self.parser.add_argument("--data", type=str, default="ODIR-5k")
        self.parser.add_argument("--image_size", type=int, default=256, help="image size")

    def get_args(self):
        args = self.parser.parse_args()
        return args
