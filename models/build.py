from torchvision.models import resnet18, resnet50
import torch.nn as nn


class build_model(nn.Module):
    def __init__(self, args):
        super(build_model, self).__init__()
        type(self).__name__ = "MyModel"
        resnet = nn.Sequential(*list(resnet50(weights="DEFAULT").children())[:-2])

        cos_loss = nn.CosineSimilarity().cuda()

        self.add_module("resnet", resnet)
        self.add_module("cos_loss", cos_loss)

    def forward(self, x):
        return self.resnet(x)
