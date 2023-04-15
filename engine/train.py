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
import torch.nn as nn
import os
from sklearn.metrics import roc_auc_score

from backpack import backpack, extend
from backpack.extensions import BatchGrad
from collections import OrderedDict

from data.dataset import get_data_weights, get_batch

from utils.info import terminal_msg
from utils.model import count_parameters, save_checkpoint, resume_checkpoint
from utils.metrics import multi_label_metrics, single_label_metrics, roc_auc_score, accuracy_score, binary_metrics
from models.build import build_single_task_model

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

        elif self.args.data in ["Kaggle", "APTOS", "DDR", "IDRiD"]:
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


class Single_Task_Trainer_maod(object):
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
                ma_img = sample['ma_img']
                od_img = sample['od_img']
                img, gt = img.to(self.device, non_blocking=True), gt.to(self.device, non_blocking=True)
                ma_img, od_img = ma_img.to(self.device, non_blocking=True), od_img.to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    pred, loss = self.model.process(img, gt, ma_img, od_img)
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
                ma_img = sample['ma_img']
                od_img = sample['od_img']
                img, gt = img.to(self.device), gt.to(self.device)
                ma_img, od_img = ma_img.to(self.device), od_img.to(self.device)
                if self.args.multi_gpus:
                    pred, _ = self.model.module.process(img, gt, ma_img, od_img)
                else:
                    pred, _ = self.model.process(img, gt, ma_img, od_img)
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

                if self.args.accumulate:
                    for i in range(10):
                        sample = get_batch(data = data)
                        img = sample['image']
                        gt = sample['landmarks']
                        img, gt = img.to(self.device, non_blocking=True), gt.to(self.device, non_blocking=True)
                        with torch.cuda.amp.autocast():
                            pred, loss = self.model.process(img, gt, roll)
                            loss = loss / 10
                            self.model.backward(loss, scaler)
                    scaler.step(self.model.optimizer)
                    scaler.update()
                    self.model.optimizer.zero_grad()
                
                else:
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
                if self.args.use_wandb:
                    wandb.log({"epoch": epoch})
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

class Multi_Task_Trainer_v2(object):
    '''
    For Gradient methods use
    '''
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
                
                '''normal loss backpropagation
                all_loss = 0
                all_loss_with_grad = 0
                for count_data in data_dict:
                    roll = count_data
                    data = self.train_data[roll]
                    #print(roll)
                    #print("\rSelect dataset {} in this batch".format(roll))

                    sample = get_batch(data = data)
                    img = sample['image']
                    gt = sample['landmarks']
                    img, gt = img.to(self.device, non_blocking=True), gt.to(self.device, non_blocking=True)
                    with torch.cuda.amp.autocast():
                        pred, loss = self.model.process(img, gt, roll)
                        loss = loss / 10
                        #self.model.backward(loss, scaler)
                        all_loss += loss.item()
                        all_loss_with_grad += loss
                        if count_data == data_dict[-1]:
                            self.model.backward(all_loss_with_grad, scaler)
                scaler.step(self.model.optimizer)
                scaler.update()
                self.model.optimizer.zero_grad()
                '''
                
                # GradVac
                '''
                grad_index = []
                for param in self.model.encoder.parameters():
                    grad_index.append(param.data.numel())
                grad_dim = sum(grad_index)

                all_loss = 0
                losses = {}

                for roll in data_dict:
                    data = self.train_data[roll]

                    sample = get_batch(data = data)
                    img = sample['image']
                    gt = sample['landmarks']
                    img, gt = img.to(self.device, non_blocking=True), gt.to(self.device, non_blocking=True)

                    pred, loss = self.model.process(img, gt, roll)
                    #loss = loss / 10
                    losses[roll] = loss
                    all_loss += loss.item()
                    
                    grads = torch.zeros(10, grad_dim).to('cuda')

                for i in data_dict:
                    losses[i].backward(losses[i].clone().detach(), retain_graph=True) if i!='DR+' else losses[i].backward(losses[i].clone().detach())
                    
                    grad = torch.zeros(grad_dim)
                    count = 0
                    for param in self.model.encoder.parameters():
                        if param.grad is not None:
                            beg = 0 if count == 0 else sum(grad_index[:count])
                            end = sum(grad_index[:(count+1)])
                            grad[beg:end] = param.grad.data.view(-1)
                        count += 1
                    grads[data_dict.index(i)] = grad
                    self.model.encoder.zero_grad()
                
                rho_T = torch.zeros(10, 10).to('cuda')

                beta = 0.5

                batch_weight = np.ones(len(losses))
                pc_grads = grads.clone()
                for i in data_dict:
                    tn_i = data_dict.index(i)
                    task_index = list(range(10))
                    task_index.remove(tn_i)
                    random.shuffle(task_index)
                    for tn_j in task_index:
                        rho_ij = torch.dot(pc_grads[tn_i], grads[tn_j]) / (pc_grads[tn_i].norm()*grads[tn_j].norm())
                        if rho_ij < rho_T[tn_i, tn_j]:
                            w = pc_grads[tn_i].norm()*(rho_T[tn_i, tn_j]*(1-rho_ij**2).sqrt()-rho_ij*(1-rho_T[tn_i, tn_j]**2).sqrt())/(grads[tn_j].norm()*(1-rho_T[tn_i, tn_j]**2).sqrt())
                            pc_grads[tn_i] += grads[tn_j]*w
                            batch_weight[tn_j] += w.item()
                            rho_T[tn_i, tn_j] = (1-beta)*rho_T[tn_i, tn_j] + beta*rho_ij
                new_grads = pc_grads.sum(0)
                count = 0
                
                # reset grad
                for param in self.model.encoder.parameters():
                    if param.grad is not None:
                        beg = 0 if count == 0 else sum(grad_index[:count])
                        end = sum(grad_index[:(count+1)])
                        param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
                    count += 1
                
                self.model.optimizer.step()
                '''

                # CAGrad
                grad_index = []
                for param in self.model.encoder.parameters():
                    grad_index.append(param.data.numel())
                grad_dim = sum(grad_index)

                all_loss = 0
                losses = {}

                for roll in data_dict:
                    data = self.train_data[roll]

                    sample = get_batch(data = data)
                    img = sample['image']
                    gt = sample['landmarks']
                    img, gt = img.to(self.device, non_blocking=True), gt.to(self.device, non_blocking=True)

                    pred, loss = self.model.process(img, gt, roll)
                    #loss = loss / 10
                    losses[roll] = loss
                    all_loss += loss.item()
                    
                    grads = torch.zeros(10, grad_dim).to('cuda')

                for i in data_dict:
                    losses[i].backward(losses[i].clone().detach(), retain_graph=True) if i!='DR+' else losses[i].backward(losses[i].clone().detach())
                    
                    grad = torch.zeros(grad_dim)
                    count = 0
                    for param in self.model.encoder.parameters():
                        if param.grad is not None:
                            beg = 0 if count == 0 else sum(grad_index[:count])
                            end = sum(grad_index[:(count+1)])
                            grad[beg:end] = param.grad.data.view(-1)
                        count += 1
                    grads[data_dict.index(i)] = grad
                    self.model.encoder.zero_grad()
                
                calpha=0.5
                rescale=1
                from scipy.optimize import minimize

                GG = torch.matmul(grads, grads.t()).cpu() # [num_tasks, num_tasks]
                g0_norm = (GG.mean()+1e-8).sqrt() # norm of the average gradient

                x_start = np.ones(10) / 10
                bnds = tuple((0,1) for x in x_start)
                cons=({'type':'eq','fun':lambda x:1-sum(x)})
                A = GG.numpy()
                b = x_start.copy()
                c = (calpha*g0_norm+1e-8).item()
                def objfn(x):
                    return (x.reshape(1,-1).dot(A).dot(b.reshape(-1,1))+c*np.sqrt(x.reshape(1,-1).dot(A).dot(x.reshape(-1,1))+1e-8)).sum()
                res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
                w_cpu = res.x
                ww = torch.Tensor(w_cpu).to('cuda')
                gw = (grads * ww.view(-1, 1)).sum(0)
                gw_norm = gw.norm()
                lmbda = c / (gw_norm+1e-8)
                g = grads.mean(0) + lmbda * gw
                if rescale == 0:
                    new_grads = g
                elif rescale == 1:
                    new_grads = g / (1+calpha**2)
                elif rescale == 2:
                    new_grads = g / (1 + calpha)
                else:
                    raise ValueError('No support rescale type {}'.format(rescale))

                
                # reset grad
                count = 0
                for param in self.model.encoder.parameters():
                    if param.grad is not None:
                        beg = 0 if count == 0 else sum(grad_index[:count])
                        end = sum(grad_index[:(count+1)])
                        param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
                    count += 1
                
                self.model.optimizer.step()
                self.model.optimizer.zero_grad()

                # Determine approximate time left
                batch_done = epoch * self.batches + batch
                batches_left = self.epochs * self.batches - batch_done
                time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # Print log
                sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %s" %
                                (epoch, self.epochs,
                                batch, self.batches,
                                all_loss,
                                time_left))
                # wandb
                if self.args.use_wandb:
                    wandb.log({"loss": all_loss})

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


class Multi_Task_Trainer_v3(object):
    '''
    For Domain Generalization use
    '''
    def __init__(self, args, model, device, train_data=None, valid_dataloaders=None):
        self.args = args
        self.model = model
        self.device = device
        self.batch_size = self.args.batch_size

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
        hparams={}
        hparams['lambda']=(1000., lambda r: 10**r.uniform(1., 4.))
        hparams['penalty_anneal_iters']=(1500, lambda r: int(r.uniform(0., 5000.)))
        #hparams['ema']=(0.95, lambda r: r.uniform(0.90, 0.99))
        hparams['ema']=0.95
        self.hparams = hparams
        for i in self.model.loss.keys():
            #self.model.decoder[i] = extend(self.model.decoder[i])
            self.model.loss[i] = extend(self.model.loss[i])
        #self.model.encoder = extend(self.model.encoder)
        self.model.decoder = extend(self.model.decoder)
        
        # copyied from github.com/alexrame/fishr/coloredmnist/train_coloredmnist.py
        def l2_between_dicts(dict_1, dict_2):
            assert len(dict_1) == len(dict_2)
            dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
            dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
            return (
                torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
                torch.cat(tuple([t.view(-1) for t in dict_2_values]))
            ).pow(2).mean()

        self.model.args.mode = 'train'
        best_precision = 0
        save_best = False
        prev_time = time.time()
        terminal_msg(f"Params in {type(self.model).__name__}: {self.model.num_params / 1e6:.4f}M ({self.model.num_trainable_params / 1e6:.4f}M trainable). "+"Start training...", 'E')

        data_dict = self.args.data.split(", ") # ['ODIR-5K', 'TAOP', 'RFMiD']
        self.num_domains = 10

        self.ema_per_domain = {}
        for data in data_dict:
            self.ema_per_domain[data] = MovingAverage(ema=self.hparams["ema"], oneminusema_correction=True)

        for epoch in range(self.start_epoch, self.epochs + 1):
            self.model.train()
            scaler = torch.cuda.amp.GradScaler()
            for batch in range(self.batches):

                grads_var_per_domain = {}
                all_loss = 0

                for count_data in data_dict:
                    roll = count_data
                    data = self.train_data[roll]
                    #print(roll)
                    #print("\rSelect dataset {} in this batch".format(roll))

                    sample = get_batch(data = data)
                    img = sample['image']
                    gt = sample['landmarks']
                    img, gt = img.to(self.device, non_blocking=True), gt.to(self.device, non_blocking=True)

                    head = roll
                    #representation = self.model.encoder(img)
                    pred, loss = self.model.process(img, gt, head)

                    #self.optimizer.zero_grad()
                    all_loss += loss

                    with backpack(BatchGrad()):
                        loss.backward(
                            inputs=list(self.model.decoder.parameters()), retain_graph=True, create_graph=True
                        )

                    # compute individual grads for all samples across all domains simultaneously
                    dict_grads = OrderedDict(
                        [
                            (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                            for name, weights in self.model.decoder.named_parameters()
                        ]
                    )

                    # grads var per domain
                    grads_var_per_domain[roll] = {}
                    for name, _grads in dict_grads.items():
                        _mean = _grads.mean(dim=0, keepdim=True)
                        _grads_centered = _grads - _mean
                        grads_var_per_domain[roll][name] = (_grads_centered).pow(2).mean(dim=0)

                # moving average
                for domain in data_dict:
                    grads_var_per_domain[domain] = self.ema_per_domain[domain].update(
                        grads_var_per_domain[domain]
                    )
                    
                # compute gradient variances averaged across domains
                grads_var = OrderedDict(
                    [
                        (
                            name,
                            torch.stack(
                                [
                                    grads_var_per_domain[domain][name]
                                    for domain in data_dict
                                ],
                                dim=0
                            ).mean(dim=0)
                        )
                        for name in grads_var_per_domain['TAOP'].keys()
                    ]
                )

                penalty = 0
                for domain in data_dict:
                    penalty += l2_between_dicts(grads_var_per_domain[domain], grads_var)
                penalty = penalty / self.num_domains

                penalty_weight = 1e8

                objective = all_loss/10 + penalty_weight * penalty

                self.model.optimizer.zero_grad()
                objective.backward()
                self.model.optimizer.step()

                # Determine approximate time left
                batch_done = epoch * self.batches + batch
                batches_left = self.epochs * self.batches - batch_done
                time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # Print log
                sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] [penalty: %f] ETA: %s" %
                                (epoch, self.epochs,
                                batch, self.batches,
                                objective,
                                penalty_weight * penalty,
                                time_left))
                # wandb
                if self.args.use_wandb:
                    wandb.log({"loss": objective})

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

class MovingAverage:

    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.named_parameters = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.named_parameters[name]

            # print(self.ema)
            # print(previous_data.shape)
            # print(data.shape)
            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.named_parameters[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data
    
class Multi_Task_Trainer_with_Domain_Discriminator(object):
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
                pred, loss, g_loss, d_loss = self.model.process(img, gt, roll)

                # Determine approximate time left
                batch_done = epoch * self.batches + batch
                batches_left = self.epochs * self.batches - batch_done
                time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # Print log
                sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [g_loss: %f] [d_loss: %f] ETA: %s" %
                                (epoch, self.epochs,
                                batch, self.batches,
                                g_loss.item(),
                                d_loss.item(),
                                time_left))
                # wandb
                if self.args.use_wandb:
                    wandb.log({"loss ({})".format(roll): loss.item(),
                               "d_loss ({})".format(roll): d_loss.item(),
                               "g_loss ({})".format(roll): g_loss.item()})

            # save best model
            if epoch % self.valid_freq == 0:
                precision = self.validate()
                if self.args.use_wandb:
                    wandb.log({"epoch": epoch})
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
                    pred = self.model.process_without_grad(img, gt, valid_dataloader_name)
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
    

class Multi_Task_Trainer_with_Multiple_Domain_Discriminator(object):
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
                pred, loss, g_loss, d_loss = self.model.process(img, gt, roll)

                # Determine approximate time left
                batch_done = epoch * self.batches + batch
                batches_left = self.epochs * self.batches - batch_done
                time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # Print log
                sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [g_loss: %f] [d_loss: %f] ETA: %s" %
                                (epoch, self.epochs,
                                batch, self.batches,
                                g_loss.item(),
                                d_loss[roll].item(),
                                time_left))
                # wandb
                if self.args.use_wandb:
                    wandb.log({"loss ({})".format(roll): loss.item(),
                               "d_loss ({})".format(roll): d_loss[roll].item(),
                               "g_loss ({})".format(roll): g_loss.item()})

            # save best model
            if epoch % self.valid_freq == 0:
                precision = self.validate()
                if self.args.use_wandb:
                    wandb.log({"epoch": epoch})
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
                    pred = self.model.process_without_grad(img, gt, valid_dataloader_name)
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


class Multi_Task_Trainer_Pseudo(object):
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

                if self.args.accumulate:
                    for i in range(10):
                        sample = get_batch(data = data)
                        img = sample['image']
                        gt = sample['landmarks']
                        img, gt = img.to(self.device, non_blocking=True), gt.to(self.device, non_blocking=True)
                        with torch.cuda.amp.autocast():
                            pred, loss = self.model.process(img, gt, roll)
                            loss = loss / 10
                            self.model.backward(loss, scaler)
                    scaler.step(self.model.optimizer)
                    scaler.update()
                    self.model.optimizer.zero_grad()
                
                else:
                    sample = get_batch(data = data)
                    img = sample['image']
                    #gt = sample['landmarks']
                    path = '/home/hssun/thesis/archive/checkpoints/' + roll +'/model_best.pth'
                    checkpoint = torch.load(path)
                    pseudo_args = self.args
                    pseudo_args.data = roll
                    single_model = build_single_task_model(pseudo_args)
                    single_model.encoder.load_state_dict(checkpoint['encoder'])
                    single_model.decoder.load_state_dict(checkpoint['decoder'])
                    with torch.no_grad():
                        pred = single_model(img)
                        if roll in ["TAOP", "APTOS", "Kaggle", "DDR"]:
                            pred = torch.argmax(pred, dim = 1)
                        elif roll in ["AMD", "LAG", "PALM", "REFUGE"]:
                            pred = pred[:, 0]

                    img = img.to(self.device, non_blocking=True)
                    gt = pred.to(self.device)
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
                if self.args.use_wandb:
                    wandb.log({"epoch": epoch})
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

class Multi_Task_Trainer_maod(object):
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

                if self.args.accumulate:
                    for i in range(10):
                        sample = get_batch(data = data)
                        img = sample['image']
                        gt = sample['landmarks']
                        img, gt = img.to(self.device, non_blocking=True), gt.to(self.device, non_blocking=True)
                        with torch.cuda.amp.autocast():
                            pred, loss = self.model.process(img, gt, roll)
                            loss = loss / 10
                            self.model.backward(loss, scaler)
                    scaler.step(self.model.optimizer)
                    scaler.update()
                    self.model.optimizer.zero_grad()
                
                else:
                    sample = get_batch(data = data)
                    img = sample['image']
                    gt = sample['landmarks']
                    ma_img = sample['ma_img']
                    od_img = sample['od_img']
                    img, gt = img.to(self.device, non_blocking=True), gt.to(self.device, non_blocking=True)
                    ma_img, od_img = ma_img.to(self.device, non_blocking=True), od_img.to(self.device, non_blocking=True)
                    with torch.cuda.amp.autocast():
                        pred, loss = self.model.process(img, gt, ma_img, od_img, roll)
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
                if self.args.use_wandb:
                    wandb.log({"epoch": epoch})
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
                    ma_img = sample['ma_img']
                    od_img = sample['od_img']
                    img, gt = img.to(self.device), gt.to(self.device)
                    ma_img, od_img = ma_img.to(self.device), od_img.to(self.device)
                    pred, _ = self.model.process(img, gt, ma_img, od_img, valid_dataloader_name)
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