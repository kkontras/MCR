import os
import sys

import torch
from fusions.common_fusions import Concat
from training_structures.cca_onestage import test, train
from unimodals.common_models import LeNet, Linear

from datasets.avmnist.get_data import get_dataloader
from utils.helper_modules import Sequential2

sys.path.append(os.getcwd())


traindata, validdata, testdata = get_dataloader("/home/pliang/yiwei/avmnist/_MFAS/avmnist", batch_size=800)
channels = 6
encoders = [
    LeNet(1, channels, 3).cuda(),
    Sequential2(LeNet(1, channels, 5), Linear(192, 48, xavier_init=True)).cuda(),
]
# encoders=[MLP(300,512,outdim), MLP(4096,1024,outdim)]
# encoders=[MLP(300, 512, 512), VGG16(512)]
# encoders=[Linear(300, 512), Linear(4096,512)]
# head=MLP(2*outdim,2*outdim,23).cuda()
head = Linear(96, 10, xavier_init=True).cuda()
fusion = Concat().cuda()

train(
    encoders,
    fusion,
    head,
    traindata,
    validdata,
    25,
    outdim=48,
    save="best_cca.pt",
    optimtype=torch.optim.AdamW,
    lr=1e-2,
)
# ,weight_decay=0.01)

print("Testing:")
model = torch.load("best_cca.pt").cuda()
test(model, testdata)
