import os
import sys

import torch
from fusions.common_fusions import LowRankTensorFusion
from private_test_scripts.all_in_one import all_in_one_test, all_in_one_train
from training_structures.Simple_Late_Fusion import test, train
from unimodals.common_models import MLP, LeNet

from datasets.avmnist.get_data import get_dataloader

sys.path.append(os.getcwd())


filename = "lowrank.pt"
traindata, validdata, testdata = get_dataloader("/data/yiwei/avmnist/_MFAS/avmnist")
channels = 6
encoders = [
    LeNet(1, channels, 3).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
    LeNet(1, channels, 5).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
]
head = MLP(channels * 20, 100, 10).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

fusion = LowRankTensorFusion([channels * 8, channels * 32], channels * 20, 40).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


def trpr():
    train(
        encoders,
        fusion,
        head,
        traindata,
        validdata,
        30,
        optimtype=torch.optim.SGD,
        lr=0.05,
        weight_decay=0.0002,
        save=filename,
    )


all_in_one_train(trpr, [encoders[0], encoders[1], fusion, head])
print("Testing:")
model = torch.load(filename).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


def tepr():
    test(model, testdata)


all_in_one_test(tepr, [model])
