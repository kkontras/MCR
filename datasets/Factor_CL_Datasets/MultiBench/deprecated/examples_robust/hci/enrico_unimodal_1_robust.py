import os
import sys

import torch
from robustness.all_in_one import general_test, general_train
from training_structures.unimodal import test, train
from unimodals.common_models import Linear, VGG11Slim

from datasets.enrico.get_data import get_dataloader
from datasets.enrico.get_data_robust import get_dataloader_robust

sys.path.append(os.getcwd())


dls, weights = get_dataloader("datasets/enrico/dataset")
traindata, validdata, _ = dls
robustdata = get_dataloader_robust("datasets/enrico/dataset", img_noise=False)
modalnum = 1
encoder = VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).cuda()
head = Linear(16, 20).cuda()
# head = MLP(16, 32, 20, dropout=False).cuda()

allmodules = [encoder, head]


def trainprocess(filename_encoder, filename_head):
    train(
        encoder,
        head,
        traindata,
        validdata,
        50,
        optimtype=torch.optim.Adam,
        lr=0.0001,
        weight_decay=0,
        modalnum=modalnum,
        save_encoder=filename_encoder,
        save_head=filename_head,
    )


filename = general_train(trainprocess, "enrico_unimodal_1", encoder=True)


def testprocess(encoder, head, testdata):
    return test(encoder, head, testdata, modalnum=modalnum)


general_test(testprocess, filename, robustdata, encoder=True)
