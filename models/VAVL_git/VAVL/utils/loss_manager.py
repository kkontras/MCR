import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


class LogManager:
    def __init__(self):
        self.log_book = dict()

    def alloc_stat_type(self, stat_type):
        self.log_book[stat_type] = []

    def alloc_stat_type_list(self, stat_type_list):
        for stat_type in stat_type_list:
            self.alloc_stat_type(stat_type)

    def init_stat(self):
        for stat_type in self.log_book.keys():
            self.log_book[stat_type] = []

    def add_stat(self, stat_type, stat):
        assert stat_type in self.log_book, "Wrong stat type"
        self.log_book[stat_type].append(stat)

    def add_torch_stat(self, stat_type, stat):
        assert stat_type in self.log_book, "Wrong stat type"
        self.log_book[stat_type].append(stat.detach().cpu().item())

    def get_stat(self, stat_type):
        result_stat = 0
        stat_list = self.log_book[stat_type]
        if len(stat_list) != 0:
            result_stat = np.mean(stat_list)
            result_stat = np.round(result_stat, 4)
        return result_stat

    def print_stat(self):
        for stat_type in self.log_book.keys():
            if len(self.log_book[stat_type]) == 0:
                continue
            stat = self.get_stat(stat_type)
            print(stat_type, ":", stat, end=" / ")
        print(" ")


def MSE_loss(pred, lab):
    loss = nn.MSELoss()
    output = loss(pred, lab)
    return output


# For Hard-label learning
def CE_category(pred, lab):
    celoss = nn.CrossEntropyLoss()
    if len(lab[0]) > 1:
        max_indx = torch.argmax(lab, dim=1)
        # print(max_indx)
    ce_loss = celoss(pred, max_indx)
    return ce_loss


def calc_err(pred, lab):
    p = pred.detach()
    t = lab.detach()
    total_num = p.size()[0]
    ans = torch.argmax(p, dim=1)
    tar = torch.argmax(t, dim=1)
    corr = torch.sum((ans == tar).long())
    err = (total_num - corr) / total_num
    return err


def CCC_loss(pred, lab, m_lab=None, v_lab=None):
    """
    pred: (N, 3)
    lab: (N, 3)
    """
    m_pred = torch.mean(pred, 0, keepdim=True)
    m_lab = torch.mean(lab, 0, keepdim=True)

    d_pred = pred - m_pred
    d_lab = lab - m_lab

    v_pred = torch.var(pred, 0, unbiased=False)
    v_lab = torch.var(lab, 0, unbiased=False)

    corr = torch.sum(d_pred * d_lab, 0) / (torch.sqrt(torch.sum(d_pred**2, 0)) * torch.sqrt(torch.sum(d_lab**2, 0)))

    s_pred = torch.std(pred, 0, unbiased=False)
    s_lab = torch.std(lab, 0, unbiased=False)

    ccc = (2 * corr * s_pred * s_lab) / (v_pred + v_lab + (m_pred[0] - m_lab[0]) ** 2)
    return ccc


def calc_acc(pred, lab):
    err = calc_err(pred, lab)
    return 1.0 - err


def scores(root):

    preds = root + "y_pred.csv"
    truths = root + "y_true.csv"

    df_preds = pd.read_csv(preds)
    df_truths = pd.read_csv(truths)

    columns = ["0", "1", "2", "3", "4", "5"]
    test_preds = df_preds[columns].values.tolist()
    test_truth = df_truths[columns].values.tolist()

    predictions, truths = [], []
    # print(test_preds, 'tpreds')

    def softmax(x):
        return np.exp(x) / sum(np.exp(x))

    for i in range(len(test_preds)):
        x = np.argmax(softmax(test_preds[i]))
        predictions.append(x)
    # print(predictions, 'preds')

    for i in range(len(test_truth)):
        x = np.argmax(test_truth[i])
        truths.append(x)

    test_preds = predictions
    test_truth = truths

    f1ma = f1_score(test_truth, test_preds, average="macro")
    f1mi = f1_score(test_truth, test_preds, average="micro")
    pre_ma = precision_score(test_truth, test_preds, average="macro")
    pre_mi = precision_score(test_truth, test_preds, average="micro")
    re_ma = recall_score(test_truth, test_preds, average="macro")
    re_mi = recall_score(test_truth, test_preds, average="micro")

    conf_matrix = confusion_matrix(test_truth, test_preds)

    # print the confusion matrix
    print(conf_matrix)

    print("F1-Score Macro = {:5.3f}".format(f1ma))
    print("F1-Score Micro = {:5.3f}".format(f1mi))
    print("-------------------------")
    print("Precision Macro = {:5.3f}".format(pre_ma))
    print("Precision Micro = {:5.3f}".format(pre_mi))
    print("-------------------------")
    print("Recall Macro = {:5.3f}".format(re_ma))
    print("Recall Micro = {:5.3f}".format(re_mi))
    print("-------------------------")
