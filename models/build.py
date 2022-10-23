from .encoder_decoder import Encoder, Decoder
import torch.nn as nn
import torch


class build_model(nn.Module):
    def __init__(self, args):
        super(build_model, self).__init__()
        type(self).__name__ = "MyModel"
        self.encoder = Encoder()
        if args.data == "ODIR-5k":
            self.decoder = Decoder(num_clas=8)
        elif args.data == "RFMiD":
            self.decoder = Decoder(num_clas=46)

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

        return pred, loss

    def backward(self, loss=None):
        loss.backward()
        self.optimizer.step()
