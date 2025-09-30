import os
import sys

import torch
from training_structures.architecture_search import test

from datasets.mimic.get_data import get_dataloader

sys.path.append(os.getcwd())

traindata, validdata, testdata = get_dataloader(1, imputed_path="datasets/mimic/im.pk")

model = torch.load("temp/best.pt").cuda()
test(model, testdata, auprc=True)
