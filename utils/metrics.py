from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, recall_score, cohen_kappa_score, accuracy_score
import numpy as np
from termcolor import colored


def Multi_AUC_and_Kappa(pred_list, gt_list):
    temp = []
    temp2 = []
    pred_list = np.array(pred_list > 0.5, dtype=float)
    #print(roc_auc_score(gt_list[:, 0], pred_list[:, 0], average='micro', sample_weight=None))
    for i in range(gt_list.shape[1]):
        auc = roc_auc_score(gt_list[:, i], pred_list[:, i], average='micro', sample_weight=None)
        kappa = cohen_kappa_score(gt_list[:, i], pred_list[:, i])
    temp.append(auc)
    temp2.append(kappa)
    return np.mean(temp), np.mean(temp2)


def multi_label_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            }


def single_label_metrics(pred, target):
    print(colored("Accuracy: ", "red") + str(accuracy_score(target, pred)))
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            }
