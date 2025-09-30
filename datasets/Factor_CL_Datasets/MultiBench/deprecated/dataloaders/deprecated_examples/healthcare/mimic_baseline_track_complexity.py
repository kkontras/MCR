import os
import sys

import torch
from fusions.common_fusions import Concat
from private_test_scripts.all_in_one import all_in_one_test, all_in_one_train
from training_structures.Simple_Late_Fusion import test, train
from unimodals.common_models import GRU, MLP

from datasets.mimic.get_data import get_dataloader

sys.path.append(os.getcwd())
# get dataloader for icd9 classification task 7
traindata, validdata, testdata = get_dataloader(7, imputed_path="datasets/mimic/im.pk")

# build encoders, head and fusion layer
encoders = [MLP(5, 10, 10, dropout=False).cuda(), GRU(12, 30, dropout=False).cuda()]
head = MLP(730, 40, 2, dropout=False).cuda()
fusion = Concat().cuda()
allmodules = [encoders[0], encoders[1], head, fusion]

# train


def trainprocess():
    train(encoders, fusion, head, traindata, validdata, 20, auprc=True)


all_in_one_train(trainprocess, allmodules)


# test
print("Testing: ")
model = torch.load("best.pt").cuda()


def testprocess():
    test(model, testdata, auprc=True)


all_in_one_test(testprocess, [model])
