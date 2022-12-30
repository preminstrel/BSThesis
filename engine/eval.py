import time
import datetime
import wandb
import sys
import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
from termcolor import colored

from utils.info import terminal_msg
from utils.model import count_parameters, save_checkpoint, resume_checkpoint
from utils.metrics import multi_label_metrics, single_label_metrics, binary_metrics

class Single_Task_Evaluation(object):
    def __init__(self, args, model, device, valid_dataloader=None):
        self.args = args
        self.model = model
        self.device = device

        model = model.to(self.device)

        self.valid_dataloader = valid_dataloader
        self.batch_size = self.args.batch_size

        if self.args.resume:
            resume_checkpoint(self, self.args.resume)
        else:
            terminal_msg("Please define ckpt!", "F")

        self.eval()
    
    def eval(self):
        terminal_msg("Start Evaluation...")
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
            
        if self.args.data in ["ODIR-5K", "RFMiD", "DR+"]:
            avg_auc, avg_kappa, avg_f1 = multi_label_metrics(gt_list, pred_list)
            print(colored("Avg AUC, Avg Kappa, Avg F1 Socre: ", "red"), (avg_auc, avg_kappa, avg_f1))

        elif self.args.data in ["Kaggle", "APTOS", "DDR"]:
            acc, kappa = single_label_metrics(gt_list, pred_list)
            print(colored("Acc, Quadratic Weighted Kappa: ", "red"), (acc, kappa))

        elif self.args.data == "TAOP":
            acc = accuracy_score(gt_list, pred_list)
            print(colored("Acc: ", "red"), acc)
        
        elif self.args.data in ["AMD", "LAG", "PALM", "REFUGE"]:
            auc, kappa, f1 = binary_metrics(gt_list, pred_list)
            print(colored("AUC, Kappa, F1 Socre: ", "red"), (auc, kappa, f1))

        terminal_msg("Evaluation phase finished!", "C")

class Multi_Task_Evaluation(object):
    def __init__(self, args, model, device, valid_dataloaders=None):
        self.args = args
        self.model = model
        self.device = device

        model = model.to(self.device)

        self.valid_dataloaders = valid_dataloaders
        self.batch_size = self.args.batch_size

        if self.args.resume:
            resume_checkpoint(self, self.args.resume)
        else:
            terminal_msg("Please define ckpt!", "F")

        self.eval()
    
    def eval(self):
        score = []
        terminal_msg("Start Evaluation...")
        threshold = 0.5
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
                    if self.args.data in ["TAOP", "Kaggle", "APTOS"]:
                        gt = [item for sublist in gt for item in sublist]
                        gt = [int(x) for x in gt]

                    pred_list.extend(pred)
                    gt_list.extend(gt)
                pred_list = np.array(pred_list)
                gt_list = np.array(gt_list)
                
            if valid_dataloader_name in ["ODIR-5K", "DR+", "RFMiD"]:
                avg_auc, avg_kappa, avg_f1 = multi_label_metrics(gt_list, pred_list)
                print(colored("Avg AUC, Avg Kappa, Avg F1 Socre: ", "red"), (avg_auc, avg_kappa, avg_f1))
                score.append(np.mean([avg_auc, avg_kappa, avg_f1]))

            elif valid_dataloader_name in ["Kaggle", "APTOS", "DDR"]:
                acc, kappa = single_label_metrics(gt_list, pred_list)
                print(colored("Acc, Quadratic Weighted Kappa: ", "red"), (acc, kappa))
                score.append(np.mean([acc, kappa]))
            
            elif valid_dataloader_name == "TAOP":
                acc = accuracy_score(gt_list, pred_list)
                print(colored("Acc: ", "red"), acc)
                score.append(acc)
        
            elif valid_dataloader_name in ["AMD", "LAG", "PALM", "REFUGE"]:
                auc, kappa, f1 = binary_metrics(gt_list, pred_list)
                print(colored("AUC, Kappa, F1 Socre: ", "red"), (auc, kappa, f1))
                score.append(np.mean([auc, kappa, f1]))
        print("Final Score: ", np.array(score).mean())
        terminal_msg("Evaluation phase finished!", "C")
