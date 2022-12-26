from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, recall_score, cohen_kappa_score, accuracy_score, precision_recall_curve, roc_curve
import numpy as np
from termcolor import colored

def multi_label_metrics(gt_list, pred_list):
    avg_auc_list = []
    avg_kappa_list = []
    avg_f1_list = []

    for i in range(gt_list.shape[1]):
        auc = roc_auc_score(gt_list[:, i], pred_list[:, i])
        precision, recall, thresholds = precision_recall_curve(gt_list[:, i], pred_list[:, i])
        kappa_list = []
        f1_list = []
        for threshold in thresholds:
            y_scores = pred_list[:, i]
            y_scores = np.array(y_scores >= threshold, dtype=float)
            kappa = cohen_kappa_score(gt_list[:, i], y_scores)
            kappa_list.append(kappa)
            f1 = f1_score(gt_list[:, i], y_scores)
            f1_list.append(f1)
        #print("AUC for case {}:".format(i), auc, ", Kappa for case {}:".format(i), kappa)
        kappa_f1 = np.array(kappa_list) + np.array(f1_list)
        avg_auc_list.append(auc)
        avg_kappa_list.append(kappa_list[np.argmax(kappa_f1)])
        avg_f1_list.append(f1_list[np.argmax(kappa_f1)])
    return np.mean(avg_auc_list), np.mean(avg_kappa_list), np.mean(avg_f1_list)

def single_label_metrics(gt_list, pred_list):
    acc = accuracy_score(gt_list, pred_list)
    kappa = cohen_kappa_score(gt_list, pred_list, weights="quadratic")
    return acc, kappa

def binary_metrics(gt_list, pred_list):
    auc = roc_auc_score(gt_list, pred_list)
    precision, recall, thresholds = precision_recall_curve(gt_list, pred_list)
    kappa_list = []
    f1_list = []
    for threshold in thresholds:
        y_scores = pred_list
        y_scores = np.array(y_scores >= threshold, dtype=float)
        kappa = cohen_kappa_score(gt_list, y_scores)
        kappa_list.append(kappa)
        f1 = f1_score(gt_list, y_scores)
        f1_list.append(f1)
    kappa_f1 = np.array(kappa_list) + np.array(f1_list)
    
    return auc, kappa_list[np.argmax(kappa_f1)], f1_list[np.argmax(kappa_f1)]
