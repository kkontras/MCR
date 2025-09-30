import os
import sys

import torch
from fusions.MVAE import ProductOfExperts
from objective_functions.recon import elbo_loss, sigmloss1d
from private_test_scripts.all_in_one import all_in_one_test, all_in_one_train
from training_structures.MVAE_mixed import test_MVAE, train_MVAE
from unimodals.common_models import MLP
from unimodals.MVAE import MLPEncoder, TSDecoder, TSEncoder

from datasets.mimic.get_data import get_dataloader

sys.path.append(os.getcwd())


traindata, validdata, testdata = get_dataloader(7, imputed_path="datasets/mimic/im.pk", flatten_time_series=True)
classes = 2
n_latent = 200
series_dim = 12
timestep = 24
fuse = ProductOfExperts((1, 40, n_latent))
encoders = [
    MLPEncoder(5, 20, n_latent).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
    TSEncoder(series_dim, 30, n_latent, timestep).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
]
decoders = [
    MLP(n_latent, 20, 5).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
    TSDecoder(series_dim, 30, n_latent, timestep).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
]
head = MLP(n_latent, 20, classes).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
elbo = elbo_loss([sigmloss1d, sigmloss1d], [1.0, 1.0], 0.0)


def trpr():
    train_MVAE(encoders, decoders, head, fuse, traindata, validdata, elbo, 30)


all_in_one_train(trpr, [encoders[0], encoders[1], decoders[0], decoders[1], head])
mvae = torch.load("best1.pt")
head = torch.load("best2.pt")


def tepr():
    test_MVAE(mvae, head, testdata)


all_in_one_test(tepr, [encoders[0], encoders[1], head])
