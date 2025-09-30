from datasets.Factor_CL_Datasets.MultiBench.datasets.affect.get_data import get_dataloader
from datasets.Factor_CL_Datasets.MultiBench.FactorCL.multibench_model import *


class FactorCL_Dataloader:

    def __init__(self, config):
        """
        :param config:
        """
        self.config = config

        g = torch.Generator()
        g.manual_seed(0)

        train_loader, valid_loader, test_loader = get_dataloader(
            self.config.dataset.data_roots,
            robust_test=False,
            batch_size=32,
            train_shuffle=True,
        )

        train_loader.generator = g
        train_loader.worker_init_fn = lambda worker_id: np.random.seed(15 + worker_id)

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
