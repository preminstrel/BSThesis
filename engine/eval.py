import time
import datetime
import wandb
import sys
import torch
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score
from termcolor import colored

from utils.info import terminal_msg
from utils.model import count_parameters, save_checkpoint, resume_checkpoint
from utils.metrics import Multi_AUC, multi_label_metrics, single_label_metrics

class Evaluation(object):
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
                print(gt)
                print(pred)
                exit()
            pred_list = np.array(pred_list)
            gt_list = np.array(gt_list)
            
        if self.args.data in ["ODIR-5K", "RFMiD"]:
            result = multi_label_metrics(pred_list, gt_list, threshold=0.5)
            print(colored("Micro F1 Score: ", "red") + str(result['micro/f1']) + colored(", Macro F1 Score: ", "red") + str(result['macro/f1']) + colored(", Samples F1 Score: ", "red") + str(result['samples/f1']))

        elif self.args.data in ["TAOP"]:
            result = single_label_metrics(pred_list, gt_list)
            print(colored("Micro F1 Score: ", "red") + str(result['micro/f1']) + colored(", Macro F1 Score: ", "red") + str(result['macro/f1']))

        terminal_msg("Evaluation phase finished!", "C")