import os
import sys

import torch
from private_test_scripts.all_in_one import all_in_one_test, all_in_one_train
from training_structures.unimodal import test, train
from unimodals.common_models import Linear, VGG11Slim

from datasets.enrico.get_data import get_dataloader

sys.path.append(os.getcwd())


dls, weights = get_dataloader("datasets/enrico/dataset")
traindata, validdata, testdata = dls
modalnum = 0
encoder = VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).cuda()
head = Linear(16, 20).cuda()
# head = MLP(16, 32, 20, dropout=False).cuda()

allmodules = [encoder, head]


def trainprocess():
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
    )


all_in_one_train(trainprocess, allmodules)

print("Testing:")
model = torch.load("best.pt").cuda()


def testprocess():
    test(encoder, head, testdata, modalnum=modalnum)


all_in_one_test(testprocess, [model])
