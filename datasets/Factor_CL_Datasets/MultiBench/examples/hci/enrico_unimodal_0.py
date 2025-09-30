import os
import sys

import torch

sys.path.append(os.getcwd())

from fusions.common_fusions import Concat  # noqa
from memory_profiler import memory_usage  # noqa
from private_test_scripts.all_in_one import all_in_one_test, all_in_one_train  # noqa
from training_structures.unimodal import test, train  # noqa
from unimodals.common_models import Linear, VGG11Slim  # noqa

from datasets.enrico.get_data import get_dataloader  # noqa

dls, weights = get_dataloader("datasets/enrico/dataset")
traindata, validdata, testdata = dls
modalnum = 0
encoder = VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).cuda()
head = Linear(16, 20).cuda()
# head = MLP(16, 32, 20, dropout=False).cuda()

allmodules = [encoder, head]


def trainprocess():
    train(
        encoder,
        head,
        traindata,
        validdata,
        50,
        optimtype=torch.optim.Adam,
        lr=0.0001,
        weight_decay=0,
        modalnum=modalnum,
    )


all_in_one_train(trainprocess, allmodules)


model = torch.load("best.pt").cuda()
test(encoder, head, testdata, dataset="enrico", modalnum=modalnum)
