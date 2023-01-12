# models.py
# created by Sylvestre-Alvise Rebuffi [srebuffi@robots.ox.ac.uk]
# Copyright © The University of Oxford, 2017-2020
# This code is made available under the Apache v2.0 licence, see LICENSE.txt for details

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
import config_task

from .encoder_decoder import Decoder_multi_classification, Decoder_single_classification

def get_task_head(input = 256, data="TAOP, APTOS, DDR, AMD, LAG, PALM, REFUGE, ODIR-5K, RFMiD, DR+"):
    decoder = {}
    if "ODIR-5K" in data:
        decoder['7'] = Decoder_multi_classification(num_class = 8, input=input)
    if "RFMiD" in data:
        decoder['8'] = Decoder_multi_classification(num_class = 29, input=input)
    if "DR+" in data:
        decoder['9'] = Decoder_multi_classification(num_class = 28, input=input)
    if "TAOP" in data:
        decoder['0'] = Decoder_single_classification(num_class = 5, input=input)
    if "APTOS" in data:
        decoder['1'] = Decoder_single_classification(num_class = 5, input=input)
    if "AMD" in data:
        decoder['3'] = Decoder_multi_classification(num_class = 1, input=input)
    if "DDR" in data:
        decoder['2'] = Decoder_single_classification(num_class = 6, input=input)
    if "LAG" in data:
        decoder['4'] = Decoder_multi_classification(num_class = 1, input=input)
    if "PALM" in data:
        decoder['5'] = Decoder_multi_classification(num_class = 1, input=input)
    if "REFUGE" in data:
        decoder['6'] = Decoder_multi_classification(num_class = 1, input=input)
    return decoder

def get_task_loss(data):
    loss = {}
    if "ODIR-5K" in data:
        loss['7'] = nn.BCEWithLogitsLoss()
    if "RFMiD" in data:
        loss['8'] = nn.BCEWithLogitsLoss()
    if "DR+" in data:
        loss['9'] = nn.BCEWithLogitsLoss()
    if "TAOP" in data:
        loss['0'] = nn.CrossEntropyLoss()
    if "APTOS" in data:
        loss['1'] = nn.CrossEntropyLoss()
    if "AMD" in data:
        loss['3'] = nn.BCEWithLogitsLoss()
    if "DDR" in data:
        loss['2'] = nn.CrossEntropyLoss()
    if "LAG" in data:
        loss['4'] = nn.BCEWithLogitsLoss()
    if "PALM" in data:
        loss['5'] = nn.BCEWithLogitsLoss()
    if "REFUGE" in data:
        loss['6'] = nn.BCEWithLogitsLoss()
    return loss

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1_fonc(in_planes, out_planes=None, stride=1, bias=False):
    if out_planes is None:
        return nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=stride, padding=0, bias=bias)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)

class conv1x1(nn.Module):
    
    def __init__(self, planes, out_planes=None, stride=1):
        super(conv1x1, self).__init__()
        if config_task.mode == 'series_adapters':
            self.conv = nn.Sequential(nn.BatchNorm2d(planes), conv1x1_fonc(planes))
        elif config_task.mode == 'parallel_adapters':
            self.conv = conv1x1_fonc(planes, out_planes, stride) 
        else:
            self.conv = conv1x1_fonc(planes)
    def forward(self, x):
        y = self.conv(x)
        if config_task.mode == 'series_adapters':
            y += x
        return y

class conv_task(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1, nb_tasks=1, is_proj=1, second=0):
        super(conv_task, self).__init__()
        self.is_proj = is_proj
        self.second = second
        self.conv = conv3x3(in_planes, planes, stride)
        if config_task.mode == 'series_adapters' and is_proj:
            self.bns = nn.ModuleList([nn.Sequential(conv1x1(planes), nn.BatchNorm2d(planes)) for i in range(nb_tasks)])
        elif config_task.mode == 'parallel_adapters' and is_proj:
            self.parallel_conv = nn.ModuleList([conv1x1(in_planes, planes, stride) for i in range(nb_tasks)])
            self.bns = nn.ModuleList([nn.BatchNorm2d(planes) for i in range(nb_tasks)])
        else:
            self.bns = nn.ModuleList([nn.BatchNorm2d(planes) for i in range(nb_tasks)])
    
    def forward(self, x):
        task = config_task.task
        y = self.conv(x)
        if self.second == 0:
            if config_task.isdropout1:
                x = F.dropout2d(x, p=0.5, training = self.training)
        else:
            if config_task.isdropout2:
                x = F.dropout2d(x, p=0.5, training = self.training)
        if config_task.mode == 'parallel_adapters' and self.is_proj:
            y = y + self.parallel_conv[task](x)
        y = self.bns[task](y)

        return y

# No projection: identity shortcut
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut=0, nb_tasks=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_task(in_planes, planes, stride, nb_tasks, is_proj=int(config_task.proj[0]))
        self.conv2 = nn.Sequential(nn.ReLU(True), conv_task(planes, planes, 1, nb_tasks, is_proj=int(config_task.proj[1]), second=1))
        self.shortcut = shortcut
        if self.shortcut == 1:
            self.avgpool = nn.AvgPool2d(2)
        
    def forward(self, x):
        residual = x
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut == 1:
            residual = self.avgpool(x)
            residual = torch.cat((residual, residual*0),1)
        y += residual
        y = F.relu(y)
        return y


class ResNet(nn.Module):
    def __init__(self, data, block, nblocks):
        super(ResNet, self).__init__()
        type(self).__name__ = "Adapter"
        nb_tasks = 10
        blocks = [block, block, block]
        factor = config_task.factor
        self.in_planes = int(32*factor)
        self.pre_layers_conv = conv_task(3,int(32*factor), 1, nb_tasks) 
        self.layer1 = self._make_layer(blocks[0], int(64*factor), nblocks[0], stride=2, nb_tasks=nb_tasks)
        self.layer2 = self._make_layer(blocks[1], int(128*factor), nblocks[1], stride=2, nb_tasks=nb_tasks)
        self.layer3 = self._make_layer(blocks[2], int(256*factor), nblocks[2], stride=2, nb_tasks=nb_tasks)
        self.end_bns = nn.ModuleList([nn.Sequential(nn.BatchNorm2d(int(256*factor)),nn.ReLU(True)) for i in range(nb_tasks)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linears = get_task_head(data=data)
        for i in self.linears:
            self.add_module(str(i), self.linears[i])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _make_layer(self, block, planes, nblocks, stride=1, nb_tasks=1):
        shortcut = 0
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = 1
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut, nb_tasks=nb_tasks))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes, nb_tasks=nb_tasks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre_layers_conv(x)
        task = config_task.task
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.end_bns[task](x)
        #print(x.shape)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        x = self.linears[str(task)](x)
        return x


class ResNet_new(nn.Module):
    def __init__(self, data, block, nblocks):
        super(ResNet_new, self).__init__()
        type(self).__name__ = "Adapter"
        nb_tasks = 10
        blocks = [block, block, block, block]
        factor = 2
        self.in_planes = int(32*factor)
        self.pre_layers_conv = conv_task(3,int(32*factor), 1, nb_tasks) 
        self.layer1 = self._make_layer(blocks[0], int(64*factor), nblocks[0], stride=2, nb_tasks=nb_tasks)
        self.layer2 = self._make_layer(blocks[1], int(128*factor), nblocks[1], stride=2, nb_tasks=nb_tasks)
        self.layer3 = self._make_layer(blocks[2], int(256*factor), nblocks[2], stride=2, nb_tasks=nb_tasks)
        self.layer4 = self._make_layer(blocks[3], int(512*factor), nblocks[3], stride=2, nb_tasks=nb_tasks)
        #self.layer5 = self._make_layer(blocks[4], int(1024*factor), nblocks[4], stride=2, nb_tasks=nb_tasks)
        self.end_bns = nn.ModuleList([nn.Sequential(nn.BatchNorm2d(int(512*factor)),nn.ReLU(True)) for i in range(nb_tasks)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linears = get_task_head(input=1024, data=data)
        
        for i in self.linears:
            self.add_module(str(i), self.linears[i])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _make_layer(self, block, planes, nblocks, stride=1, nb_tasks=1):
        shortcut = 0
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = 1
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut, nb_tasks=nb_tasks))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes, nb_tasks=nb_tasks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre_layers_conv(x)
        task = config_task.task
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.layer5(x)
        x = self.end_bns[task](x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        x = self.linears[str(task)](x)
        return x

def resnet26(data, blocks=BasicBlock):
    return  ResNet(data, blocks, [4,4,4])

def resnet50(data, blocks=BasicBlock):
    return  ResNet_new(data, blocks, [4, 4, 4, 4])
# sgd.py
# created by Sylvestre-Alvise Rebuffi [srebuffi@robots.ox.ac.uk]
# Copyright © The University of Oxford, 2017-2020
# This code is made available under the Apache v2.0 licence, see LICENSE.txt for details

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                siz = p.grad.size()
                import numpy as np
                config_task.decay3x3 = np.ones(10) * 0.00001
                config_task.decay1x1 = np.ones(10) * 0.00001
                if len(siz) > 3:
                    if siz[2] == 3:
                        weight_decay = config_task.decay3x3[config_task.task]
                    elif siz[2] == 1:
                        weight_decay = config_task.decay1x1[config_task.task]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                
                p.data.add_(-group['lr'], d_p)

        return loss
