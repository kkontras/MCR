import os
import sys

sys.path.append(os.getcwd())

import torch
from training_structures.unimodal import test, train
from unimodals.common_models import MLP, LeNet

from datasets.avmnist.get_data import get_dataloader

modalnum = 0
traindata, validdata, testdata = get_dataloader("/data/yiwei/avmnist/_MFAS/avmnist")
channels = 3
# encoders=[LeNet(1,channels,3).cuda(),LeNet(1,channels,5).cuda()]
encoder = LeNet(1, channels, 3).cuda()
head = MLP(channels * 8, 100, 10).cuda()


train(
    encoder,
    head,
    traindata,
    validdata,
    20,
    optimtype=torch.optim.SGD,
    lr=0.01,
    weight_decay=0.0001,
    modalnum=modalnum,
)

print("Testing:")
encoder = torch.load("encoder.pt").cuda()
head = torch.load("head.pt")
test(encoder, head, testdata, modalnum=modalnum, no_robust=True)
