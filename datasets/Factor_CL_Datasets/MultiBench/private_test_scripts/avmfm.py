import os
import sys

import torch
from fusions.common_fusions import Concat
from objective_functions.recon import recon_weighted_sum, sigmloss1dcentercrop
from private_test_scripts.all_in_one import all_in_one_test, all_in_one_train
from training_structures.MFM import test_MFM, train_MFM
from unimodals.common_models import MLP
from unimodals.MVAE import DeLeNet, LeNetEncoder

from datasets.avmnist.get_data import get_dataloader

sys.path.append(os.getcwd())


traindata, validdata, testdata = get_dataloader("/data/yiwei/avmnist/_MFAS/avmnist")
channels = 6

classes = 10
n_latent = 200
fuse = Concat()

encoders = [
    LeNetEncoder(1, channels, 3, n_latent, twooutput=False).cuda(),
    LeNetEncoder(1, channels, 5, n_latent, twooutput=False).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
]
decoders = [
    DeLeNet(1, channels, 3, n_latent).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
    DeLeNet(1, channels, 5, n_latent).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
]

intermediates = [
    MLP(n_latent, n_latent // 2, n_latent // 2).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
    MLP(n_latent, n_latent // 2, n_latent // 2).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
    MLP(2 * n_latent, n_latent, n_latent // 2).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
]
head = MLP(n_latent // 2, 40, classes).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
recon_loss = recon_weighted_sum([sigmloss1dcentercrop(28, 34), sigmloss1dcentercrop(112, 130)], [1.0, 1.0])


def trpr():
    train_MFM(
        encoders,
        decoders,
        head,
        intermediates,
        fuse,
        recon_loss,
        traindata,
        validdata,
        25,
    )


all_in_one_train(trpr, encoders + decoders + intermediates + [head])
model = torch.load("best.pt")


def tepr():
    test_MFM(model, testdata)


all_in_one_test(tepr, encoders + intermediates + [head])
