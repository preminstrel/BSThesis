from .encoder_decoder import Encoder, Decoder
import torch.nn as nn
import torch
from utils.info import terminal_msg


class build_model(nn.Module):
    def __init__(self, args):
        super(build_model, self).__init__()
        self.encoder = Encoder()
        if args.data == "ODIR-5K":
            self.decoder = Decoder(num_clas=8)
            type(self).__name__ = "ODIR-5K"
        elif args.data == "RFMiD":
            self.decoder = Decoder(num_clas=46)
            type(self).__name__ = "RFMiD"
        else:
            terminal_msg("Args.Data Error", "F")
            exit()

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters())+list(self.decoder.parameters()), lr=args.lr, betas=(0.5, 0.999))

        self.add_module("encoder", self.encoder)
        self.add_module("decoder", self.decoder)
        self.add_module("bce_loss", self.bce_loss)

    def forward(self, img):
        return self.decoder(self.encoder(img))

    def process(self, img, gt):
        pred = self(img)
        self.optimizer.zero_grad()
        loss = self.bce_loss(pred, gt)

        pred = torch.sigmoid(pred)

        return pred, loss

    def backward(self, loss=None):
        loss.backward()
        self.optimizer.step()
