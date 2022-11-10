import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import resnet50

from utils.info import terminal_msg


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Encoder(nn.Module):
    '''
    ResNet-50 backbone: [B, 3, 256, 256] --> [B, 2048, 8, 8]
    '''

    def __init__(self):
        super().__init__()
        # downsample = nn.Sequential(
        #     nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(2048),
        # )
        # bottleneck = Bottleneck(2048, 512, 2, downsample)
        resnet = resnet50(weights="DEFAULT")
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        x = self.resnet(x)
        return x


class Decoder_multi_classification(nn.Module):
    '''
    Multi label classification Decoder:
    AvgPool --> FC (Sigmoid)
    '''

    def __init__(self, num_class=36):
        super(Decoder_multi_classification, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, num_class)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x


class Decoder_single_classification(nn.Module):
    '''
    Single label classification Decoder:
    AvgPool --> FC
    '''

    def __init__(self, num_class=36):
        super(Decoder_single_classification, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, num_class)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


def get_task_head(data):
    decoder = {}
    if "ODIR-5K" in data:
        decoder["ODIR-5K"] = Decoder_multi_classification(num_class=8)
        decoder["ODIR-5K"].__name__ = "ODIR-5K"
    if "RFMiD" in data:
        decoder["RFMiD"] = Decoder_multi_classification(num_class=29)
        decoder["RFMiD"].__name__ = "RFMiD"
    if "TAOP" in data:
        decoder["TAOP"] = Decoder_single_classification(num_class=5)
        decoder["TAOP"].__name__ = "TAOP"
    if "APTOS" in data:
        decoder["APTOS"] = Decoder_single_classification(num_class=5)
        decoder["APTOS"].__name__ = "APTOS"
    if "Kaggle" in data:
        decoder["Kaggle"] = Decoder_single_classification(num_class=5)
        decoder["Kaggle"].__name__ = "Kaggle"
    return decoder


def get_task_loss(data):
    loss = {}
    if "ODIR-5K" in data:
        weights = torch.tensor([2.2477, 3.8860, 21.4524, 22.2862, 24.7333, 35.8352, 26.0620, 5.5568]).cuda()
        loss["ODIR-5K"] = nn.BCELoss(weight=weights)
    if "RFMiD" in data:
        weights = torch.tensor([1.264, 5.106, 19.2, 6.057, 13.913, 19.01, 26.301, 10.323, 137.143, 40.851, 128.0, 51.892, 6.809, 68.571, 320.0,
                               120.0, 29.538, 33.103, 384.0, 112.941, 174.545, 137.143, 44.651, 60.0, 128.0, 87.273, 174.545, 320.0, 56.471]).cuda()
        loss["RFMiD"] = nn.BCELoss()
    if "TAOP" in data:
        loss["TAOP"] = nn.CrossEntropyLoss()
    if "APTOS" in data:
        loss["APTOS"] = nn.CrossEntropyLoss()
    if "Kaggle" in data:
        loss["Kaggle"] = nn.CrossEntropyLoss()
    return loss
