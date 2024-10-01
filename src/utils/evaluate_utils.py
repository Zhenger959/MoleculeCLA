'''
Author: 
Date: 2024-03-05 17:12:54
LastEditors: 
LastEditTime: 2024-06-04 10:05:35
Description: 
'''
from sklearn.metrics import roc_auc_score
import numpy as np
from src.utils import RankedLogger
log = RankedLogger(__name__, rank_zero_only=True)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calc_rocauc_score(labels, preds, valid, print_log = False):
    """compute ROC-AUC and averaged across tasks"""
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
        preds = preds.reshape(-1, 1)
        valid = valid.reshape(-1, 1)

    rocauc_list = []
    for i in range(labels.shape[1]):
        c_valid = valid[:, i].astype("bool")
        c_label, c_pred = labels[c_valid, i], preds[c_valid, i]
        #AUC is only defined when there is at least one positive data.
        if len(np.unique(c_label)) == 2:
            rocauc_list.append(roc_auc_score(c_label, c_pred))
    if print_log:
        log.info('Valid ratio: %s' % (np.mean(valid)))
        log.info('Task evaluated: %s/%s' % (len(rocauc_list), labels.shape[1]))
    if len(rocauc_list) == 0:
        # raise RuntimeError("No positively labeled data available. Cannot compute ROC-AUC.")
        return 0.0

    return sum(rocauc_list)/len(rocauc_list)

def calc_rmse(labels, preds):
    """tbd"""
    return np.sqrt(np.mean((preds - labels) ** 2))

def calc_mae(labels, preds):
    """tbd"""
    return np.mean(np.abs(preds - labels))

class Evaluator(object):
    def __init__(self, metric) -> None:
        self.metric = metric

    def __call__(self,preds, labels, valids, print_log = False):

        if self.metric == 'rmse':
            return calc_rmse(labels.detach().cpu().float().numpy(), preds.detach().cpu().numpy())
        elif self.metric == 'mae':
            return calc_mae(labels.detach().cpu().float().numpy(), preds.detach().cpu().numpy())
        else:
            return calc_rocauc_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), valids.detach().cpu().numpy(), print_log)