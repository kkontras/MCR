import os
import sys

import torch
from training_structures.architecture_search import test

from datasets.mimic.get_data_robust import get_dataloader

sys.path.append(os.getcwd())


traindata, validdata, testdata, robustdata = get_dataloader(7, imputed_path="datasets/mimic/im.pk")

model = torch.load("/home/pliang/yiwei/best_icd9_70_79.pt").cuda()
test(model, testdata, auprc=True)
acc = []
print("Robustness testing:")
for noise_level in range(len(robustdata)):
    print("Noise level {}: ".format(noise_level / 10))
    acc.append(test(model, robustdata[noise_level], auprc=False))

print("Accuracy of different noise levels:", acc)
