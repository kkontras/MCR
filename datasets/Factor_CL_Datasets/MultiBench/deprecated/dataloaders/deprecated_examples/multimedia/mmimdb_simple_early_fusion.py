import os
import sys

import torch
from fusions.common_fusions import Concat
from training_structures.Simple_Early_Fusion import test, train
from unimodals.common_models import MaxOut_MLP

from datasets.imdb.get_data import get_dataloader

sys.path.append(os.getcwd())


traindata, validdata, testdata = get_dataloader("../video/multimodal_imdb.hdf5", vgg=True, batch_size=128)

encoders = None
# encoders=[MLP(300, 512, 512), MLP(4096, 1000, 512)]
# encoders=[MLP(300, 512, 512), VGG16(512)]
head = MaxOut_MLP(23, 512, 4396).cuda()
fusion = Concat().cuda()

train(
    encoders,
    fusion,
    head,
    traindata,
    validdata,
    1000,
    early_stop=True,
    task="multilabel",
    regularization=False,
    save="best_ef.pt",
    optimtype=torch.optim.AdamW,
    lr=4e-2,
    weight_decay=0.01,
    criterion=torch.nn.BCEWithLogitsLoss(),
)

print("Testing:")
model = torch.load("best_ef.pt").cuda()
test(model, testdata, criterion=torch.nn.BCEWithLogitsLoss(), task="multilabel")
