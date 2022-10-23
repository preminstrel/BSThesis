import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import resnet50


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
    def __init__(self):
        super().__init__()
        downsample = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2048),
        )
        bottleneck = Bottleneck(2048, 512, 2, downsample)
        resnet = resnet50(weights="DEFAULT")
        #list(resnet.children())[-3][0] = bottleneck
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.conv = nn.Conv2d(2048, 32, 1)

    def forward(self, x):
        x = self.resnet(x)
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_clas=36):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(32*8*8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_clas)
        self.dropout = nn.Dropout(0.25)
        #self.fc2 = nn.Linear(1024, 512)
        #self.fc3 = nn.Linear(512, num_class)
        #self.dconv1 = nn.ConvTranspose2d(2048, 1024, 3, stride=2, padding=1)
        # self.dfc3 = nn.Linear(1000, 4096)
        # self.bn3 = nn.BatchNorm2d(4096)
        # self.dfc2 = nn.Linear(4096, 4096)
        # self.bn2 = nn.BatchNorm2d(4096)
        # self.dfc1 = nn.Linear(4096, 256 * 6 * 6)
        # self.bn1 = nn.BatchNorm2d(256*6*6)
        # self.upsample1 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        #x = torch.sigmoid(x)
        return x

        # x = self.dfc2(x)
        # x = F.relu(self.bn2(x))
        # x = self.dfc1(x)
        # x = F.relu(self.bn1(x))
        # x = x.view(batch_size, 256, 6, 6)
        # x = self.upsample1(x)
        # x = self.dconv5(x)
        # x = F.relu(x)
        # x = F.relu(self.dconv4(x))
        # x = F.relu(self.dconv3(x))
        # x = self.upsample1(x)
        # x = self.dconv2(x)
        # x = F.relu(x)
        # x = self.upsample1(x)
        # x = self.dconv1(x)
        # x = F.sigmoid(x)
        return x


class StandardED(nn.Module):
    def __init__(self, img_size):
        super(StandardED, self).__init__()
