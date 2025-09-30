import os
import sys

import torch
from private_test_scripts.all_in_one import all_in_one_test
from training_structures.architecture_search import test

from datasets.avmnist.get_data import get_dataloader

sys.path.append(os.getcwd())

traindata, validdata, testdata = get_dataloader("/data/yiwei/avmnist/_MFAS/avmnist", batch_size=32)
model = torch.load("temp/best.pt").to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


def testprocess():
    test(model, testdata)


all_in_one_test(testprocess, [model])
