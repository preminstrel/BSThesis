import argparse
import warnings
import os


class ParserArgs(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="PyTorch Training and Testing"
        )

        self.get_general_parser()

        # ablation exps
        # self.get_ablation_exp_args()
        # comparison exps
        # self.get_comparison_exps_args()

    def get_general_parser(self):
        self.parser.add_argument(
            "--resume",
            default="",
            type=str,
            metavar="PATH",
            help="path to latest checkpoint (format: version/path.tar)",
        )
        self.parser.add_argument(
            "--save_freq",
            default=10,
            type=int,
            help="ckpt save frequency (default: 10)",
        )
        self.parser.add_argument(
            "--n_epochs",
            default=200,
            type=int,
            help="number of total epochs to run (default: 200)",
        )
        self.parser.add_argument(
            "--start_epoch",
            default=0,
            type=int,
            help="start epoch number for resume use",
        )

        self.parser.add_argument(
            "--visualize", action="store_true", help="visualize mode, batch size = 1"
        )
        self.parser.add_argument("--test", action="store_true", help="test mode")
        self.parser.add_argument("--wandb", action="store_true", help="enable wandb")

        self.parser.add_argument(
            "--mode",
            default=0,
            type=int,
            help="0: pure RD, 1: img -> mask -> RD model, 2: gaussian filter processing, 3: new idea",
        )
        self.parser.add_argument(
            "--lamb", default=0, type=float, help="ratio for loss function"
        )

        self.parser.add_argument(
            "--data",
            default="data",
            type=str,
            help="OCT2017 or RESC",
            choices=["OCT2017", "RESC"],
        )
        self.parser.add_argument(
            "--batch_size", default=16, type=int, help="batch size, default = 16"
        )

    def get_args(self):
        args = self.parser.parse_args()
        return args
