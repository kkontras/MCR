# from unimodals.common_models import LeNet, MLP, Constant
import os
import sys

sys.path.append(os.getcwd())

import torch  # noqa
from fusions.common_fusions import Concat  # noqa
from torch import nn  # noqa

import utils.surrogate as surr  # noqa
from datasets.mimic.get_data import get_dataloader  # noqa

from .training_structures.architecture_search import test, train  # noqa

traindata, validdata, testdata = get_dataloader(1, imputed_path="datasets/mimic/im.pk")

model = torch.load("temp/best.pt").cuda()
# dataset = 'mimic mortality', 'mimic 1', 'mimic 7'
test(model, testdata, dataset="mimic 1", auprc=True)
