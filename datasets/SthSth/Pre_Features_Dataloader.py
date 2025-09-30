import multiprocessing

import numpy as np
import torch
from torch.utils.data import Dataset


class PreFeaturesDataset(Dataset):
    def __init__(self, config, train=True):
        self.config = config

        if train:
            data_path = self.config.dataset.train_data_path
        else:
            data_path = self.config.dataset.val_data_path

        self.data = torch.load(data_path)
        self.data["features"]["video"] = self.data["features"]["video"].cpu()
        self.data["features"]["flow"] = self.data["features"]["flow"].cpu()
        self.data["features"]["layout"] = self.data["features"]["layout"].cpu()

        self.norm = {}
        self.norm["video"] = {
            "mean": self.data["features"]["video"].mean(),
            "std": self.data["features"]["video"].std(),
        }
        self.norm["flow"] = {
            "mean": self.data["features"]["flow"].mean(),
            "std": self.data["features"]["flow"].std(),
        }
        self.norm["layout"] = {
            "mean": self.data["features"]["layout"].mean(),
            "std": self.data["features"]["layout"].std(),
        }

        self.norm["mean"] = torch.cat(
            [
                self.data["features"]["video"],
                self.data["features"]["flow"],
                self.data["features"]["layout"],
            ],
            dim=1,
        ).mean()
        self.norm["std"] = torch.cat(
            [
                self.data["features"]["video"],
                self.data["features"]["flow"],
                self.data["features"]["layout"],
            ],
            dim=1,
        ).std()

        self.data["labels"] = self.data["labels"].cpu()

    def __len__(self):
        return len(self.data["labels"])

    def __getitem__(self, idx):
        output = {"data": {}}
        output["data"][0] = self.data["features"]["video"][idx]
        output["data"][1] = self.data["features"]["flow"][idx]
        output["data"][2] = self.data["features"]["layout"][idx]

        if self.config.dataset.get("znorm", False):
            output["data"][0] = (output["data"][0] - self.norm["video"]["mean"]) / self.norm["video"]["std"]
            output["data"][1] = (output["data"][1] - self.norm["flow"]["mean"]) / self.norm["flow"]["std"]
            output["data"][2] = (output["data"][2] - self.norm["layout"]["mean"]) / self.norm["layout"]["std"]

        if self.config.dataset.get("cznorm", False):
            output["data"][0] = (output["data"][0] - self.norm["mean"]) / self.norm["std"]
            output["data"][1] = (output["data"][1] - self.norm["mean"]) / self.norm["std"]
            output["data"][2] = (output["data"][2] - self.norm["mean"]) / self.norm["std"]

        output["label"] = self.data["labels"][idx]

        return output


class SthSth_PreFeatureDataloader:

    def __init__(self, config):
        """
        :param config:
        """
        self.config = config

        dataset_train, dataset_val = self._get_datasets()

        g = torch.Generator()
        g.manual_seed(0)

        num_cores = multiprocessing.cpu_count() - 2
        # num_cores = 16
        print("We are changing dataloader workers to num of cores {}".format(num_cores))

        print("Train {}, Val {}".format(len(dataset_train), len(dataset_val)))

        self.train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.config.training_params.batch_size,
            num_workers=num_cores,
            shuffle=True,
            generator=g,
            pin_memory=self.config.training_params.pin_memory,
            worker_init_fn=lambda worker_id: np.random.seed(15 + worker_id),
        )
        self.valid_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=self.config.training_params.test_batch_size,
            shuffle=False,
            num_workers=num_cores,
            pin_memory=self.config.training_params.pin_memory,
        )

    def _get_datasets(self):

        train_dataset = PreFeaturesDataset(config=self.config, train=True)
        valid_dataset = PreFeaturesDataset(config=self.config, train=False)

        return train_dataset, valid_dataset
