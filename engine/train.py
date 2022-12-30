import time
import datetime
import wandb
import sys
import torch
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score
from termcolor import colored
from torch.nn import DataParallel
import os
from sklearn.metrics import roc_auc_score

from data.dataset import get_data_weights, get_batch

from utils.info import terminal_msg
from utils.model import count_parameters, save_checkpoint, resume_checkpoint
from utils.metrics import multi_label_metrics, single_label_metrics, roc_auc_score, accuracy_score, binary_metrics


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #np.random.seed(seed)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Single_Task_Trainer(object):
    def __init__(self, args, model, device, train_dataloader=None, valid_dataloader=None):
        self.args = args
        self.model = model
        self.device = device

        model = model.to(self.device)
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
            scaler = torch.cuda.amp.GradScaler()
            for i, sample in enumerate(self.train_dataloader):
                img = sample['image']
                gt = sample['landmarks']
                img, gt = img.to(self.device, non_blocking=True), gt.to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    pred, loss = self.model.process(img, gt)
                    self.model.backward(loss, scaler)

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

            # save best model
            if epoch % self.valid_freq == 0:
                precision = self.validate()
                if precision > best_precision:
                    best_precision = precision
                    save_best = True
                    save_checkpoint(self, epoch, save_best)
                else:
                    save_best = False

            # save ckpt for fixed freq
            if epoch % self.save_freq == 0 or epoch == self.epochs:
                save_checkpoint(self, epoch, False)
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
                if self.args.multi_gpus:
                    pred, _ = self.model.module.process(img, gt)
                else:
                    pred, _ = self.model.process(img, gt)
                pred = pred.cpu().tolist()
                gt = gt.cpu().tolist()
                if self.args.data in ["TAOP"]:
                    gt = [item for sublist in gt for item in sublist]
                    gt = [int(x) for x in gt]

                pred_list.extend(pred)
                gt_list.extend(gt)
            pred_list = np.array(pred_list, dtype=np.float32)
            gt_list = np.array(gt_list, dtype=np.float32)
            
        if self.args.data in ["ODIR-5K", "RFMiD", "DR+"]:
            avg_auc, avg_kappa, avg_f1 = multi_label_metrics(gt_list, pred_list)
            print(colored("Avg AUC, Avg Kappa, Avg F1 Socre: ", "red"), (avg_auc, avg_kappa, avg_f1))
            if self.args.use_wandb:
                wandb.log({"Avg AUC ({})".format(self.args.data): avg_auc,
                           "Avg Kappa ({})".format(self.args.data): avg_kappa,
                           "Avg F1 Score ({})".format(self.args.data): avg_f1,})
            precision = np.mean([avg_auc, avg_kappa, avg_f1])

        elif self.args.data in ["Kaggle", "APTOS", "DDR"]:
            acc, kappa = single_label_metrics(gt_list, pred_list)
            print(colored("Acc, Quadratic Weighted Kappa: ", "red"), (acc, kappa))

            if self.args.use_wandb:
                wandb.log({"Acc ({})".format(self.args.data): acc,
                            "Kappa ({})".format(self.args.data): kappa,})
            precision = np.mean([acc, kappa])

        elif self.args.data == "TAOP":
            acc = accuracy_score(gt_list, pred_list)
            print(colored("Acc: ", "red"), acc)

            if self.args.use_wandb:
                wandb.log({"Acc ({})".format(self.args.data): acc,})
            precision = acc

        elif self.args.data in ["AMD", "LAG", "PALM", "REFUGE"]:
            auc, kappa, f1 = binary_metrics(gt_list, pred_list)
            print(colored("AUC, Kappa, F1 Socre: ", "red"), (auc, kappa, f1))
            if self.args.use_wandb:
                wandb.log({
                    "AUC ({})".format(self.args.data): auc,
                    "Kappa ({})".format(self.args.data): kappa,
                    "F1 Score ({})".format(self.args.data): f1,
                })
            precision = np.mean([auc, kappa, f1])

        terminal_msg("Validation finished!", "C")
        return precision


class Multi_Task_Trainer(object):
    def __init__(self, args, model, device, train_data=None, valid_dataloaders=None):
        self.args = args
        self.model = model
        self.device = device

        model = model.to(self.device)

        self.train_data = train_data
        self.valid_dataloaders = valid_dataloaders

        self.epochs = self.args.epochs
        self.batches = self.args.batches
        self.save_freq = self.args.save_freq
        self.valid_freq = self.args.valid_freq

        if self.args.resume:
            resume_checkpoint(self, self.args.resume)
        else:
            self.start_epoch = 1

        if self.args.preflight:
            print(colored("[preflight] ", "cyan") + "Testing ckpt function...")
            save_checkpoint(self, 0, True)
            resume_checkpoint(self, f"archive/checkpoints/{args.method}/model_best.pth")
            terminal_msg("Save and resume function well!", "C")
            print(colored("[preflight] ", "cyan") + "Testing validation function...")
            self.validate()
            print(colored("[preflight] ", "cyan") + "Safe Flight!")

        self.train()

    def train(self):
        self.model.args.mode = 'train'
        best_precision = 0
        save_best = False
        prev_time = time.time()
        terminal_msg(f"Params in {type(self.model).__name__}: {self.model.num_params / 1e6:.4f}M ({self.model.num_trainable_params / 1e6:.4f}M trainable). "+"Start training...", 'E')

        data_dict = self.args.data.split(", ") # ['ODIR-5K', 'TAOP', 'RFMiD']

        for epoch in range(self.start_epoch, self.epochs + 1):
            self.model.train()
            scaler = torch.cuda.amp.GradScaler()
            for batch in range(self.batches):
                # weighted random select a dataset
                roll = random.choices(data_dict)[0]
                data = self.train_data[roll]
                #print("\rSelect dataset {} in this batch".format(roll))
                
                sample = get_batch(data = data)
                img = sample['image']
                gt = sample['landmarks']
                img, gt = img.to(self.device, non_blocking=True), gt.to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    pred, loss = self.model.process(img, gt, roll)
                    self.model.backward(loss, scaler)

                # Determine approximate time left
                batch_done = epoch * self.batches + batch
                batches_left = self.epochs * self.batches - batch_done
                time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # Print log
                sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %s" %
                                 (epoch, self.epochs,
                                  batch, self.batches,
                                  loss.item(),
                                  time_left))
                # wandb
                if self.args.use_wandb:
                    wandb.log({"loss ({})".format(roll): loss.item()})

            # save best model
            if epoch % self.valid_freq == 0:
                precision = self.validate()
                if precision > best_precision:
                    best_precision = precision
                    save_best = True
                    save_checkpoint(self, epoch, save_best)
                else:
                    save_best = False

            # save ckpt for fixed freq
            if epoch % self.save_freq == 0 or epoch == self.epochs:
                save_checkpoint(self, epoch, False)
        terminal_msg("Training phase finished!", "C")

    def validate(self):
        self.model.args.mode = 'validate'
        precision = []
        all_precision = []
        print(colored('\n[Executing]', 'blue'), 'Start validating...')
        dataloader_dict = self.args.data.split(", ")

        for roll in range(len(dataloader_dict)):
            valid_dataloader_name = dataloader_dict[roll]
            valid_dataloader = self.valid_dataloaders[valid_dataloader_name]
            print("\rSelect dataset {} to eval".format(valid_dataloader_name))

            with torch.no_grad():
                pred_list = []
                gt_list = []
                self.model.eval()
                for i, sample in enumerate(valid_dataloader):
                    img = sample['image']
                    gt = sample['landmarks']
                    img, gt = img.to(self.device), gt.to(self.device)
                    pred, _ = self.model.process(img, gt, valid_dataloader_name)
                    pred = pred.cpu().tolist()
                    gt = gt.cpu().tolist()
                    if self.args.data in ["TAOP", "APTOS", "Kaggle", "DDR", "PALM", "LAG", "AMD", "REFUGE"]:
                        gt = [item for sublist in gt for item in sublist]
                        gt = [int(x) for x in gt]

                    pred_list.extend(pred)
                    gt_list.extend(gt)

                pred_list = np.array(pred_list)
                gt_list = np.array(gt_list)

            if valid_dataloader_name in ["ODIR-5K", "DR+", "RFMiD"]:
                avg_auc, avg_kappa, avg_f1 = multi_label_metrics(gt_list, pred_list)
                print(colored("Avg AUC, Avg Kappa, Avg F1 Socre: ", "red"), (avg_auc, avg_kappa, avg_f1))
                
                if self.args.use_wandb:
                    wandb.log({"Avg AUC ({})".format(valid_dataloader_name): avg_auc,
                    "Avg Kappa ({})".format(valid_dataloader_name): avg_kappa,
                    "Avg F1 Score ({})".format(valid_dataloader_name): avg_f1,
                    })
                
                precision = np.mean(np.mean([avg_auc, avg_kappa, avg_f1]))
                all_precision.append(precision)

            elif valid_dataloader_name in ["Kaggle", "APTOS", "DDR"]:
                acc, kappa = single_label_metrics(gt_list, pred_list)
                print(colored("Acc, Quadratic Weighted Kappa: ", "red"), (acc, kappa))

                if self.args.use_wandb:
                    wandb.log({
                    "Acc ({})".format(valid_dataloader_name): acc,
                    "Kappa ({})".format(valid_dataloader_name): kappa,
                    })
                
                precision = np.mean([acc, kappa])
                all_precision.append(precision)
            
            elif valid_dataloader_name == "TAOP":
                acc = accuracy_score(gt_list, pred_list)
                print(colored("Acc: ", "red"), acc)

                if self.args.use_wandb:
                    wandb.log({"Acc ({})".format(valid_dataloader_name): acc})
                
                precision = acc
                all_precision.append(precision)

            elif valid_dataloader_name in ["AMD", "LAG", "PALM", "REFUGE"]:
                auc, kappa, f1 = binary_metrics(gt_list, pred_list)
                print(colored("AUC, Kappa, F1 Socre: ", "red"), (auc, kappa, f1))

                if self.args.use_wandb:
                    wandb.log({
                    "AUC ({})".format(valid_dataloader_name): auc,
                    "Kappa ({})".format(valid_dataloader_name): kappa,
                    "F1 Score ({})".format(valid_dataloader_name): f1,
                    })
                
                precision = np.mean([auc, kappa, f1])
                all_precision.append(precision)

        precision = np.array(all_precision).mean()
        print(colored("Final Score: ", "red"), precision)
        if self.args.use_wandb:
            wandb.log({"Final Score": precision})
        terminal_msg("Validation finished!", "C")

        return precision
