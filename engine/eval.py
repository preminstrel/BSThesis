import time
import datetime
import wandb
import sys
import torch
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from termcolor import colored

from utils.info import terminal_msg
from utils.model import count_parameters, save_checkpoint, resume_checkpoint
from utils.metrics import Multi_AUC_and_Kappa, multi_label_metrics, single_label_metrics


class Single_Task_Evaluation(object):
    def __init__(self, args, model, device, valid_dataloader=None):
        self.args = args
        self.model = model
        self.device = device

        model = model.to(self.device)
        if len(self.args.device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.args.device_ids)

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
                # print(gt)
                # print(pred)
                # exit()
            pred_list = np.array(pred_list)
            gt_list = np.array(gt_list)

        if self.args.data in ["ODIR-5K", "RFMiD"]:
            result = multi_label_metrics(pred_list, gt_list, threshold=0.5)
            print(colored("AUC, Kappa: ", "red") + str(Multi_AUC_and_Kappa(pred_list, gt_list)))
            print(colored("Micro F1 Score: ", "red") + str(result['micro/f1']) + colored(", Macro F1 Score: ", "red") +
                  str(result['macro/f1']) + colored(", Samples F1 Score: ", "red") + str(result['samples/f1']))

        elif self.args.data in ["TAOP"]:
            result = single_label_metrics(pred_list, gt_list)
            print(colored("Micro F1 Score: ", "red") + str(result['micro/f1']) + colored(", Macro F1 Score: ", "red") + str(result['macro/f1']))

        terminal_msg("Evaluation phase finished!", "C")


class Multi_Task_Evaluation(object):
    def __init__(self, args, model, device, valid_dataloaders=None):
        self.args = args
        self.model = model
        self.device = device

        model = model.to(self.device)
        if len(self.args.device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.args.device_ids)

        self.valid_dataloaders = valid_dataloaders
        self.batch_size = self.args.batch_size

        if self.args.resume:
            resume_checkpoint(self, self.args.resume)
        else:
            terminal_msg("Please define ckpt!", "F")

        self.eval()

    def eval(self):
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
                    if self.args.data in ["TAOP"]:
                        gt = [item for sublist in gt for item in sublist]
                        gt = [int(x) for x in gt]

                    pred_list.extend(pred)
                    gt_list.extend(gt)
                    # print(gt)
                    # print(pred)
                    # exit()
                pred_list = np.array(pred_list)
                gt_list = np.array(gt_list)

            if valid_dataloader_name in ["ODIR-5K"]:
                result = multi_label_metrics(pred_list, gt_list, threshold=threshold)
                print(colored("AUC, Kappa: ", "red") + str(Multi_AUC_and_Kappa(pred_list, gt_list)))
                print(colored("Micro F1 Score: ", "red") + str(result['micro/f1']) + colored(", Macro F1 Score: ", "red") +
                      str(result['macro/f1']) + colored(", Samples F1 Score: ", "red") + str(result['samples/f1']))

            elif valid_dataloader_name in ["TAOP"]:
                result = single_label_metrics(pred_list, gt_list)
                print(colored("Micro F1 Score: ", "red") + str(result['micro/f1']) + colored(", Macro F1 Score: ", "red") + str(result['macro/f1']))

            elif valid_dataloader_name == "RFMiD":
                result = multi_label_metrics(pred_list, gt_list, threshold=threshold)
                pred_list = np.array(pred_list > 0.5, dtype=float)
                print(colored("Disease Risk AUC: ", "red") + str(roc_auc_score(gt_list[:, 0], pred_list[:, 0], average='micro', sample_weight=None)))
                print(colored("Micro F1 Score: ", "red") + str(result['micro/f1']) + colored(", Macro F1 Score: ", "red") +
                      str(result['macro/f1']) + colored(", Samples F1 Score: ", "red") + str(result['samples/f1']))

        terminal_msg("Evaluation phase finished!", "C")
