import time
import datetime
import wandb
import sys
import torch
import numpy as np
import random

from utils.info import terminal_msg
from utils.model import count_parameters, save_checkpoint, resume_checkpoint


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer(object):
    def __init__(self, args, model, device, train_dataloader=None, valid_dataloader=None):
        self.args = args
        self.model = model
        self.device = device

        model = model.to(self.device)
        if len(self.args.device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.args.device_ids)

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.epochs = self.args.epochs
        self.save_freq = self.args.save_freq
        self.valid_freq = self.args.valid_freq
        self.batch_size = self.args.batch_size

        if self.args.resume:
            resume_checkpoint(self, self.args.resume)
        else:
            self.start_epoch = 1

        self.train()

    def train(self):
        self.model.train()
        best_acc = 0
        save_best = False
        num_params, num_trainable_params = count_parameters(self.model)
        terminal_msg(f"Params in {type(self.model).__name__}: {num_params / 1e6:.4f}M ({num_trainable_params / 1e6:.4f}M trainable). "+"Start training...", 'E')

        for epoch in range(self.start_epoch, self.epochs + 1):
            for i, (img, gt) in enumerate(self.train_dataloader):
                img, gt = img.to(self.device), gt.to(self.device)

                loss, output = self.model.process(img)
                self.model.backward(output, gt)

                # Determine approximate time left
                batches_done = self.epoch * self.train_dataloader.__len__() + i
                batches_left = self.n_epochs * self.train_dataloader.__len__() - batches_done
                time_left = datetime.timedelta(
                    seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # Print log
                sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %s" %
                                 (epoch, self.epochs,
                                  i, self.train_dataloader.__len__(),
                                  loss.item(),
                                  time_left))
                # wandb
                if self.args.use_wandb:
                    wandb.log({"loss": loss.item()})

            if epoch % self.valid_freq == 0:
                self.validate()

            # save ckpt
            if epoch % self.save_freq == 0 or epoch == self.epochs:
                acc = self.validate()
                if acc > best_acc:
                    best_acc = acc
                    save_best = True
                else:
                    save_best = False
                save_checkpoint(self, epoch, save_best)
        terminal_msg("Training phase finished!", "C")

    def validate(self):
        self.model.eval()
        acc = 0
        terminal_msg("Start validation phase...", "E")
        with torch.no_grad:
            for i, (img, gt) in enumerate(self.valid_dataloader):
                img, gt = img.to(self.device), gt.to(self.device)

                output = self.model(img)

        terminal_msg("Validation phase finished!", "C")
        return acc
