import os
import sys

import torch
from private_test_scripts.Augmented_Multitask1 import test

from datasets.mimic.multitask import get_dataloader

sys.path.append(os.getcwd())

# get dataloader for icd9 classification task 7
traindata, validdata, testdata = get_dataloader(imputed_path="/home/pliang/yiwei/im.pk")


# test
print("Testing: ")
model = torch.load("best2.pt").to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
test(model, testdata)
