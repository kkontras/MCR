import os
import sys

import torch
from private_test_scripts.all_in_one import all_in_one_test
from training_structures.architecture_search import test

from datasets.mimic.get_data import get_dataloader

sys.path.append(os.getcwd())


traindata, validdata, testdata = get_dataloader(1, imputed_path="datasets/mimic/im.pk")

model = torch.load("temp/best.pt").to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


def tepr():
    test(model, testdata, auprc=True)


all_in_one_test(tepr, [model])
