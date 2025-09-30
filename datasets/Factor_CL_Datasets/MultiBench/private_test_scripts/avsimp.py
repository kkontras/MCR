import os
import sys

import torch
from fusions.common_fusions import Concat
from private_test_scripts.all_in_one import all_in_one_test
from training_structures.Simple_Late_Fusion import test, train
from unimodals.common_models import MLP, LeNet

from datasets.avmnist.get_data import get_dataloader

sys.path.append(os.getcwd())


traindata, validdata, testdata = get_dataloader("/data/yiwei/avmnist/_MFAS/avmnist")
channels = 6
encoders = [
    LeNet(1, channels, 3).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
    LeNet(1, channels, 5).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
]
head = MLP(channels * 40, 100, 10).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

fusion = Concat().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


def trainprocess():
    train(
        encoders,
        fusion,
        head,
        traindata,
        validdata,
        25,
        optimtype=torch.optim.SGD,
        lr=0.1,
        weight_decay=0.0001,
    )


# all_in_one_train(trainprocess,[encoders[0],encoders[1],head,fusion])
print("Testing:")

model = torch.load("best.pt").to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


def testprocess():
    test(model, testdata)


all_in_one_test(testprocess, [model])
