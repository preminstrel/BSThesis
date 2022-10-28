from ast import arg
from json import decoder
from .encoder_decoder import Encoder, Decoder_multi_classification, Decoder_single_classification
from .encoder_decoder import get_task_head, get_task_loss
import torch.nn as nn
import torch

from utils.info import terminal_msg, get_device
from utils.model import count_parameters


class build_single_task_model(nn.Module):
    '''
    build single-task model as baselines
    '''

    def __init__(self, args):
        super(build_single_task_model, self).__init__()
        self.encoder = Encoder()
        self.args = args
        if args.data == "ODIR-5K":
            self.decoder = Decoder_multi_classification(num_class=8)
            type(self).__name__ = "ODIR-5K"
        elif args.data == "RFMiD":
            self.decoder = Decoder_multi_classification(num_class=46)
            type(self).__name__ = "RFMiD"
        elif args.data == "TAOP":
            self.decoder = Decoder_single_classification(num_class=5)
            type(self).__name__ = "TAOP"
        else:
            terminal_msg("Args.Data Error (From build_single_task_model.__init__)", "F")
            exit()

        self.bce_loss = nn.BCELoss()
        self.nll_loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters())+list(self.decoder.parameters()), lr=args.lr, betas=(0.5, 0.999))

        self.add_module("encoder", self.encoder)
        self.add_module("decoder", self.decoder)
        self.add_module("bce_loss", self.bce_loss)
        self.add_module("nll_loss", self.nll_loss)

    def forward(self, img):
        return self.decoder(self.encoder(img))

    def process(self, img, gt):
        pred = self(img)
        self.optimizer.zero_grad()
        if self.args.data in ["ODIR-5K", "RFMiD"]:
            loss = self.bce_loss(pred, gt)
        elif self.args.data in ["TAOP"]:
            gt = torch.LongTensor(gt.long().squeeze().cpu()).cuda()
            loss = self.nll_loss(pred, gt)
            pred = torch.argmax(pred, dim=1)
        else:
            terminal_msg("Error (From build_single_task_model.process)", "F")
            exit()
        return pred, loss

    def backward(self, loss=None):
        loss.backward()
        self.optimizer.step()


class build_hard_param_share_model(nn.Module):
    '''
    build hard params shared multi-task model
    '''

    def __init__(self, args):
        super(build_hard_param_share_model, self).__init__()
        self.encoder = Encoder()
        type(self).__name__ = "Hard_Params"
        self.args = args
        self.decoder = get_task_head(args.data)
        self.loss = get_task_loss(args.data)
        device = get_device()

        self.encoder.to(device)
        num_params, num_trainable_params = count_parameters(self.encoder)

        decoder_params = []
        for i in self.decoder:
            decoder_params += list(self.decoder[i].parameters())
            self.decoder[i].to(device)
            num_params_increment, num_trainable_params_increment = count_parameters(self.decoder[i])
            num_params += num_params_increment
            num_trainable_params += num_trainable_params_increment

        self.num_params = num_params
        self.num_trainable_params = num_trainable_params

        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + decoder_params, lr=args.lr, betas=(0.5, 0.999))

        self.add_module("encoder", self.encoder)
        for i in self.decoder:
            self.add_module(str(i), self.decoder[i])

    def forward(self, img, head):
        presentation = self.encoder(img)
        pred = self.decoder[head](presentation)
        return presentation, pred

    def process(self, img, gt, head):
        presentation, pred = self(img, head)
        self.optimizer.zero_grad()

        if head == "TAOP":
            gt = torch.LongTensor(gt.long().squeeze().cpu()).cuda()
            loss = self.loss[head](pred, gt)
            pred = torch.argmax(pred, dim=1)
            return pred, loss

        loss = self.loss[head](pred, gt)

        return pred, loss

    def backward(self, loss=None):
        loss.backward()
        self.optimizer.step()
