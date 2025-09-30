import os
import sys

import torch
from training_structures.unimodal import test, train
from unimodals.common_models import LeNet, Linear

from datasets.avmnist.get_data import get_dataloader

sys.path.append(os.getcwd())

traindata, validdata, testdata = get_dataloader("/data/yiwei/avmnist/_MFAS/avmnist")
channels = 3
encoders = LeNet(1, channels, 5).cuda()
head = Linear(channels * 32, 10).cuda()
mn = 1

train(
    encoders,
    head,
    traindata,
    validdata,
    100,
    optimtype=torch.optim.SGD,
    lr=0.1,
    weight_decay=0.0001,
    modalnum=mn,
)

print("Testing:")
encoder = torch.load("encoder.pt").cuda()
head = torch.load("head.pt")
test(encoder, head, testdata, modalnum=mn)
