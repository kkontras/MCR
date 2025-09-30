import os
import sys

import torch
from training_structures.architecture_search import test

from datasets.avmnist.get_data import get_dataloader

sys.path.append(os.getcwd())

traindata, validdata, testdata = get_dataloader("/data/yiwei/avmnist/_MFAS/avmnist", batch_size=32)
model = torch.load("temp/best.pt").cuda()
test(model, testdata, no_robust=True)
