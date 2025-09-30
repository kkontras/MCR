import sys

import torch
from fusions.common_fusions import MultiplicativeInteractions2Modal
from get_data_robust import get_dataloader, get_dataloader_robust
from robustness.all_in_one import general_test, general_train
from training_structures.Simple_Late_Fusion import test, train
from unimodals.common_models import Linear, MaxOut_MLP

sys.path.append("/home/pliang/multibench/MultiBench/datasets/imdb")
sys.path.append("/home/pliang/multibench/MultiBench")

# from get_data_robust import get_dataloader, get_dataloader_robust

traindata, validdata = get_dataloader("../../../video/multimodal_imdb.hdf5", batch_size=128, vgg=True)
robustdata = get_dataloader_robust("../../../video/mmimdb", "../../../video/multimodal_imdb.hdf5", batch_size=128)

encoders = [
    MaxOut_MLP(512, 512, 300, linear_layer=False),
    MaxOut_MLP(512, 1024, 4096, 512, False),
]
# encoders=[MLP(300, 512, 512), VGG16(512)]
head = Linear(1024, 23).cuda()
fusion = MultiplicativeInteractions2Modal([512, 512], 1024, "matrix").cuda()


def trainprocess(filename):
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
        save=filename,
        optimtype=torch.optim.AdamW,
        lr=8e-3,
        weight_decay=0.01,
        criterion=torch.nn.BCEWithLogitsLoss(),
    )


filename = general_train(trainprocess, "mmimdb_lf")


def testprocess(model, testdata):
    return test(model, testdata, criterion=torch.nn.BCEWithLogitsLoss(), task="multilabel")


general_test(testprocess, filename, robustdata, multi_measure=True)
