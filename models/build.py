from .encoder_decoder import Encoder, Decoder_multi_classification, Decoder_single_classification
import torch.nn as nn
import torch
from utils.info import terminal_msg


class build_model(nn.Module):
    def __init__(self, args):
        super(build_model, self).__init__()
        self.encoder = Encoder()
        self.args = args
        if args.data == "ODIR-5K":
            self.decoder = Decoder_multi_classification(num_class = 8)
            type(self).__name__ = "ODIR-5K"
        elif args.data == "RFMiD":
            self.decoder = Decoder_multi_classification(num_class = 46)
            type(self).__name__ = "RFMiD"
        elif args.data == "TAOP":
            self.decoder = Decoder_single_classification(num_class = 5)
            type(self).__name__ = "TAOP"
        else:
            terminal_msg("Args.Data Error (From build_model.__init__)", "F")
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
            pred = torch.argmax(pred, dim = 1)
        else:
            terminal_msg("Error (From buil_models.process)", "F")
            exit()
        return pred, loss

    def backward(self, loss = None):
        loss.backward()
        self.optimizer.step()
