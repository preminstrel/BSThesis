import time
import datetime
from wsgiref import validate
import wandb
import sys
import torch
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score
from termcolor import colored

from utils.info import terminal_msg
from utils.model import count_parameters, save_checkpoint, resume_checkpoint
from utils.metrics import multi_label_metrics, single_label_metrics


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

        self.validate()

        self.train()

    def train(self):
        best_precision = 0
        save_best = False
        num_params, num_trainable_params = count_parameters(self.model)
        prev_time = time.time()
        terminal_msg(f"Params in {type(self.model).__name__}: {num_params / 1e6:.4f}M ({num_trainable_params / 1e6:.4f}M trainable). "+"Start training...", 'E')

        for epoch in range(self.start_epoch, self.epochs + 1):
            self.model.train()
            for i, sample in enumerate(self.train_dataloader):
                img = sample['image']
                gt = sample['landmarks']
                img, gt = img.to(self.device), gt.to(self.device)

                pred, loss = self.model.process(img, gt)
                self.model.backward(loss)

                # Determine approximate time left
                batches_done = epoch * self.train_dataloader.__len__() + i
                batches_left = self.epochs * self.train_dataloader.__len__() - batches_done
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
                if self.valid_freq == 321:
                    save_checkpoint(self, epoch, False)
                else:
                    precision = self.validate()
                    if precision > best_precision:
                        best_precision = precision
                        save_best = True
                    else:
                        save_best = False
                    save_checkpoint(self, epoch, save_best)
        terminal_msg("Training phase finished!", "C")

    def validate(self):
        print(colored('\n[Executing]', 'blue'), 'Start validating...')
        pred_list = []
        gt_list = []
        threshold = 0
        with torch.no_grad():
            self.model.eval()
            for i, sample in enumerate(self.valid_dataloader):
                img = sample['image']
                gt = sample['landmarks']
                img, gt = img.to(self.device), gt.to(self.device)
                pred, _ = self.model.process(img, gt)
                pred = pred.cpu().tolist()
                gt = gt.cpu().tolist()
                if self.args.data in ["TAOP"]:
                    gt = [item for sublist in gt for item in sublist]
                    gt = [int(x) for x in gt]

                pred_list.extend(pred)
                gt_list.extend(gt)
            pred_list = np.array(pred_list)
            gt_list = np.array(gt_list)

        if self.args.data in ["ODIR-5K", "RFMiD"]:
            result = multi_label_metrics(pred_list, gt_list, threshold=0.5)
            print(colored("Micro F1 Score: ", "red") + str(result['micro/f1']) + colored(", Macro F1 Score: ", "red") +
                  str(result['macro/f1']) + colored(", Samples F1 Score: ", "red") + str(result['samples/f1']))
            if self.args.use_wandb:
                wandb.log({"Micro F1 Score": result['micro/f1'],
                           "Macro F1 Score": result['macro/f1'],
                           "Samples F1 Score": result['samples/f1'], })
                acc = np.mean([result['micro/f1'], result['macro/f1'], result['samples/f1']])

        elif self.args.data in ["TAOP"]:
            result = single_label_metrics(pred_list, gt_list)
            print(colored("Micro F1 Score: ", "red") + str(result['micro/f1']) + colored(", Macro F1 Score: ", "red") + str(result['macro/f1']))

            if self.args.use_wandb:
                wandb.log({"Micro F1 Score": result['micro/f1'],
                           "Macro F1 Score": result['macro/f1'], })
            acc = np.mean([result['micro/f1'], result['macro/f1']])

        terminal_msg("Validation finished!", "C")
        return acc
