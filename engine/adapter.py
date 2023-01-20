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

from data.dataset import get_data_weights, get_batch

from utils.info import terminal_msg
from utils.model import count_parameters, save_checkpoint, resume_checkpoint
from utils.metrics import multi_label_metrics, single_label_metrics, roc_auc_score, accuracy_score, binary_metrics

#from models.resnet_with_adapter import ResNet, get_task_loss, SGD, resnet26
import config_task
import models.resnet_with_adapter


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #np.random.seed(seed)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MTL_adapter(object):
    def __init__(self, args, device, train_data=None, valid_dataloaders=None):
        
        data = "TAOP, APTOS, DDR, AMD, LAG, PALM, REFUGE, ODIR-5K, RFMiD, DR+"
        model = models.resnet_with_adapter.resnet50()

        checkpoint = torch.load('/home/hssun/thesis/archive/checkpoints/HPS/baseline.pth')

        state_dict = checkpoint['encoder']
        for k in list(state_dict.keys()):
            # retain resnet
            if k.startswith('resnet'):
                if k == "resnet.0.weight":
                    state_dict['conv1.weight'] = state_dict[k]
                elif k == "resnet.1.weight":
                    state_dict['bn1.weight'] = state_dict[k]
                elif k == "resnet.1.bias":
                    state_dict['bn1.bias'] = state_dict[k]
                elif k == "resnet.1.running_mean":
                    state_dict['bn1.running_mean'] = state_dict[k]
                elif k == "resnet.1.running_var":
                    state_dict['bn1.running_var'] = state_dict[k]
                elif k == "resnet.1.num_batches_tracked":
                    state_dict['bn1.num_batches_tracked'] = state_dict[k]
                elif str(k[len("resnet.4")-1]) == '4':
                    state_dict["layer1"+k[len("resnet.")+1:]] = state_dict[k]
                    # print("layer1"+k[len("resnet.")+1:])
                elif str(k[len("resnet.5")-1]) == '5':
                    state_dict["layer2"+k[len("resnet.")+1:]] = state_dict[k]
                    # print("layer1"+k[len("resnet.")+1:])
                elif str(k[len("resnet.6")-1]) == '6':
                    state_dict["layer3"+k[len("resnet.")+1:]] = state_dict[k]
                    # print("layer1"+k[len("resnet.")+1:])
                elif str(k[len("resnet.7")-1]) == '7':
                    state_dict["layer4"+k[len("resnet.")+1:]] = state_dict[k]
                    # print("layer1"+k[len("resnet.")+1:])
                else:
                    print(k)

            # delete renamed or unused k
            del state_dict[k]
            state_dict['0.fc1.weight'] = checkpoint['TAOP']['fc1.weight']
            state_dict['0.fc1.bias'] = checkpoint['TAOP']['fc1.bias']
            state_dict['1.fc1.weight'] = checkpoint['APTOS']['fc1.weight']
            state_dict['1.fc1.bias'] = checkpoint['APTOS']['fc1.bias']
            state_dict['2.fc1.weight'] = checkpoint['DDR']['fc1.weight']
            state_dict['2.fc1.bias'] = checkpoint['DDR']['fc1.bias']
            state_dict['3.fc1.weight'] = checkpoint['AMD']['fc1.weight']
            state_dict['3.fc1.bias'] = checkpoint['AMD']['fc1.bias']
            state_dict['4.fc1.weight'] = checkpoint['LAG']['fc1.weight']
            state_dict['4.fc1.bias'] = checkpoint['LAG']['fc1.bias']
            state_dict['5.fc1.weight'] = checkpoint['PALM']['fc1.weight']
            state_dict['5.fc1.bias'] = checkpoint['PALM']['fc1.bias']
            state_dict['6.fc1.weight'] = checkpoint['REFUGE']['fc1.weight']
            state_dict['6.fc1.bias'] = checkpoint['REFUGE']['fc1.bias']
            state_dict['7.fc1.weight'] = checkpoint['ODIR-5K']['fc1.weight']
            state_dict['7.fc1.bias'] = checkpoint['ODIR-5K']['fc1.bias']
            state_dict['8.fc1.weight'] = checkpoint['RFMiD']['fc1.weight']
            state_dict['8.fc1.bias'] = checkpoint['RFMiD']['fc1.bias']
            state_dict['9.fc1.weight'] = checkpoint['DR+']['fc1.weight']
            state_dict['9.fc1.bias'] = checkpoint['DR+']['fc1.bias']

        # for i in list(state_dict.keys()):
        #     print(i)
        model.load_state_dict(state_dict, strict=False)
        for key in model.state_dict().keys():
            if "adapter" in key:
                model.state_dict()[key].data.zero_()

        for name, param in model.named_parameters():
            if "adapter" not in name and "fc" not in name:
                param.requires_grad = False
                #print(name)
        # for layer in model.children():
        #     for parameter in layer.parameters():
        #         print(layer)
        #         parameter.requires_grad = True
        #         #print(model.state_dict()[key].requires_grad)
        #         #exit()
        #         #model.state_dict()[key].requires_grad = False
        terminal_msg(f"Loaded pretrained model", "C")
        num_params, num_trainable_params = count_parameters(model)
        terminal_msg(f"Params in {type(model).__name__}: {num_params / 1e6:.4f}M ({num_trainable_params / 1e6:.4f}M trainable). "+"Start training...", 'E')

        self.args = args
        self.device = device
        self.model = model.to(self.device)

        self.train_data = train_data
        self.valid_dataloaders = valid_dataloaders

        self.epochs = self.args.epochs
        self.batches = self.args.batches
        self.save_freq = self.args.save_freq
        self.valid_freq = self.args.valid_freq
        self.loss = models.resnet_with_adapter.get_task_loss(data=data) 

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, betas=(0.5, 0.999))

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
        best_precision = 0
        save_best = False
        prev_time = time.time()
        num_params, num_trainable_params = count_parameters(self.model)
        terminal_msg(f"Params in {type(self.model).__name__}: {num_params / 1e6:.4f}M ({num_trainable_params / 1e6:.4f}M trainable). "+"Start training...", 'E')

        data_dict = self.args.data.split(", ") # ['ODIR-5K', 'TAOP', 'RFMiD']
        
        self.model.to(self.device)
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.model.train()
            scaler = torch.cuda.amp.GradScaler()
            for batch in range(self.batches):
                # weighted random select a dataset
                roll = random.choices(data_dict)[0]
                data = self.train_data[roll]
                config_task.task = data_dict.index(roll)
                #print("\rSelect dataset {} in this batch".format(roll))

                if self.args.accumulate:
                    for i in range(10):
                        sample = get_batch(data = data)
                        img = sample['image']
                        gt = sample['landmarks']
                        img, gt = img.to(self.device, non_blocking=True), gt.to(self.device, non_blocking=True)
                        with torch.cuda.amp.autocast():
                            pred = self.model(img)
                            pred, loss = self.model.process(img, gt, roll)
                            loss = loss / 10            
                            scaler.scale(loss).backward()
                            scaler.step(self.optimizer)
                            scaler.update()
                    scaler.step(self.model.optimizer)
                    scaler.update()
                    self.model.optimizer.zero_grad()
                
                else:
                    sample = get_batch(data = data)
                    img = sample['image']
                    gt = sample['landmarks']
                    img, gt = img.to(self.device, non_blocking=True), gt.to(self.device, non_blocking=True)

                    with torch.cuda.amp.autocast():
                        #pred, loss = self.model.process(img, gt, roll)
                        pred = self.model(img)

                        # print(pred.shape, gt.shape)
                        # print(roll, config_task.task)

                        if roll in ["TAOP", "APTOS", "Kaggle", "DDR"]:
                            if gt.shape[0] == 1:
                                gt = gt[0].long()
                            else:
                                gt = torch.LongTensor(gt.long().squeeze().cpu()).cuda()
                            loss = self.loss[str(config_task.task)](pred, gt)
                            pred = torch.argmax(pred, dim = 1)

                        elif roll in ["AMD", "LAG", "PALM", "REFUGE"]:
                            pred = pred[:, 0]
                            gt = gt[:, 0]
                            #print(pred, gt)
                            loss = self.loss[str(config_task.task)](pred, gt)

                        else:
                            loss = self.loss[str(config_task.task)](pred, gt)

                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()

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
                    data_dict = self.args.data.split(", ") # ['ODIR-5K', 'TAOP', 'RFMiD']
                    config_task.task = data_dict.index(valid_dataloader_name)
                    img, gt = img.to(self.device), gt.to(self.device)
                    pred = self.model(img)

                    if valid_dataloader_name in ["TAOP", "APTOS", "Kaggle", "DDR"]:
                        if gt.shape[0] == 1:
                            gt = gt[0].long()
                        else:
                            gt = torch.LongTensor(gt.long().squeeze().cpu()).cuda()
                        pred = torch.argmax(pred, dim = 1)

                    elif valid_dataloader_name in ["AMD", "LAG", "PALM", "REFUGE"]:
                        pred = pred[:, 0]
                        gt = gt[:, 0]
                        #print(pred, gt)

                    else:
                        pass
                    
                    pred = pred.cpu().tolist()
                    gt = gt.cpu().tolist()
                    # if valid_dataloader_name in ["TAOP", "APTOS", "Kaggle", "DDR", "PALM", "LAG", "AMD", "REFUGE"]:
                    #     gt = [item for sublist in gt for item in sublist]
                    #     gt = [int(x) for x in gt]

                    pred_list.extend(pred)
                    gt_list.extend(gt)

                pred_list = np.array(pred_list)
                gt_list = np.array(gt_list)

            if valid_dataloader_name in ["ODIR-5K", "DR+", "RFMiD"]:
                #print(gt_list.shape, pred_list.shape)
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