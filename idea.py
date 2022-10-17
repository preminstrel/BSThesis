import torch.nn as nn
import torch

from torchsummary import summary
from torchviz import make_dot

from models.build import build_model

from engine.train import Trainer, setup_seed

from utils.info import terminal_msg, epic_start, get_device
from utils.parser import ParserArgs
from utils.data import count_file
from utils.model import count_parameters, save_checkpoint, resume_checkpoint

from data.dataset import TrainDataset, get_transforms


if __name__ == "__main__":
    epic_start("My bachelor thesis")
    args = ParserArgs().get_args()
    setup_seed(123)
    model = build_model(args)
    device = get_device()
    #Trainer(args, model, device)
    # transfrom = get_transforms(224)
    # dataset = TrainDataset(args, transform=transfrom)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    # for i, img in enumerate(dataloader):
    #     print(img.shape)
    print(type(model).__name__)
    a, b = count_parameters(model)
    print(a, b)
