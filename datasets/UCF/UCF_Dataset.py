import copy
import os
import pickle
import random
from typing import Tuple

import einops
import librosa
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor
from torchvision import transforms


class UCF_Dataset:

    def __init__(self, config, annotation_file, data=None, mode="train"):
        self.config = config
        self.mode = mode
        self.target_audio_fps = 16000

        # self.sampling_rate = self.config.dataset.sampling_rate

        self.this_data = data
        self.return_data = config.dataset.get("return_data", {"video": True, "spectrogram": True, "audio": False})

        if self.this_data is None:
            with open(annotation_file, "r") as file:
                self.file_names = [line.replace("\n", "").strip().split(" ")[0].split("/")[-1] for line in file]

            with open(
                os.path.join(
                    self.config.dataset.data_roots,
                    "UCF_fps1_audio16000.pkl",
                ),
                "rb",
            ) as handle:
                total_data = pickle.load(handle)

            with open(os.path.join(self.config.dataset.data_roots, "class_change.pkl"), "rb") as handle:
                class_number_change = pickle.load(handle)

            self.this_data = {}
            count, skipped_noise, skipped_frame = 0, 0, 0
            for i in range(len(self.file_names)):
                if total_data[self.file_names[i]]["data"]["vision"].shape[0] < self.config.dataset.num_frame:
                    skipped_frame += 1
                    continue
                if len(total_data[self.file_names[i]]["data"]["audio"]) < 2:
                    skipped_noise += 1
                    continue
                self.this_data[count] = total_data[self.file_names[i]]
                self.this_data[count]["filename"] = self.file_names[i]
                self.this_data[count]["label"] = class_number_change[self.this_data[count]["label"]]
                count += 1

            print("We skipped {} due to no audio and {} due to less than {} frames".format(skipped_noise, skipped_frame, self.config.dataset.num_frame))

        if mode == "train":
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224, antialias=True),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(size=(224, 224), antialias=True),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        self.this_stored_data = copy.deepcopy(self.this_data)
        if mode != "test":
            self.split_dataset(mode)
        else:
            del self.this_stored_data

    def make_barplots(self, class_dict):
        num_images = []
        for idx in range(len(self.this_data)):
            num_images.append(len(self.this_data[idx]["data"]["vision"]))
        num_images = np.array(num_images)

        import matplotlib.pyplot as plt

        num, count = np.unique(num_images, return_counts=True)
        plt.bar(num, count)
        plt.title(self.mode)
        plt.show()

        labels = []
        for idx in range(len(self.this_data)):
            labels.append(self.this_data[idx]["label"])
        labels = np.array(labels)

        num, count = np.unique(labels, return_counts=True)
        plt.bar(num, count)
        plt.title(self.mode)
        plt.tight_layout()
        plt.show()

    def __len__(self):
        return len(self.this_data)

    def split_dataset(self, mode):

        indices = {"_".join(self.this_data[i]["filename"].split("_")[1:3]): self.this_data[i]["label"] for i in self.this_data}
        ids = list(indices.keys())
        labels = np.array([indices[i] for i in indices]).astype(int)

        X_train, X_val, y_train, y_val = train_test_split(
            ids,
            labels,
            test_size=self.config.dataset.get("val_split_rate", 0.1),
            random_state=self.config.training_params.seed,
            stratify=labels,
        )
        if mode == "train":
            split_data = {}
            for c, id in enumerate(X_train):
                for i in self.this_data:
                    if id in self.this_data[i]["filename"] and self.this_data[i]["filename"] not in split_data:
                        split_data[self.this_data[i]["filename"]] = self.this_data[i]

        elif mode == "val":
            split_data = {}
            for c, id in enumerate(X_val):
                for i in self.this_data:
                    if id in self.this_data[i]["filename"] and self.this_data[i]["filename"] not in split_data:
                        split_data[self.this_data[i]["filename"]] = self.this_data[i]

        split_data = {c: split_data[id] for c, id in enumerate(split_data)}
        self.this_data = split_data

    def get_frames(self, idx):
        frames = self.this_data[idx]["data"]["vision"]

        frames = frames[: self.config.dataset.num_frame]
        frames = einops.rearrange(torch.from_numpy(frames), "f h w c -> f c h w").float()

        if self.transform is not None:
            frames = self.transform(frames)
        frames = einops.rearrange(frames, "f c h w -> c f h w")
        return frames

    def get_audio(self, idx):
        if not self.return_data["audio"]:
            return False
        audio = self.this_data[idx]["data"]["audio"]
        audio[audio > 1.0] = 1.0
        audio[audio < -1.0] = -1.0

        return audio

    def get_spectrogram(self, idx):

        if not self.return_data["spectrogram"]:
            return False

        audio = self.this_data[idx]["data"]["audio"]
        audio[audio > 1.0] = 1.0
        audio[audio < -1.0] = -1.0

        spectrogram = librosa.stft(
            audio[: (self.config.dataset.num_frame - 1) * self.target_audio_fps],
            n_fft=512,
            hop_length=353,
        )
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        return spectrogram

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:

        label = self.this_data[idx]["label"]

        frames = self.get_frames(idx)
        audio = self.get_audio(idx)
        spectrogram = self.get_spectrogram(idx)

        if self.mode == "test":
            random_idx = random.randint(0, len(self.this_data) - 1)
            sh_frames = self.get_frames(random_idx)
            sh_spectrogram = self.get_audio(random_idx)

            return {
                "data": {
                    0: spectrogram,
                    1: frames,
                    "0_random_indistr": sh_spectrogram,
                    "1_random_indistr": sh_frames,
                },
                "label": label,
            }

        return {"data": {0: spectrogram, 1: frames, 2: audio}, "label": label}


class UCF_Dataloader:

    def __init__(self, config):

        self.config = config

        dataset_train, dataset_val, dataset_test = self._get_datasets()

        g = torch.Generator()
        g.manual_seed(0)

        num_cores = len(os.sched_getaffinity(0)) - 1

        print("Available cores {}".format(len(os.sched_getaffinity(0))))
        print("We are changing dataloader workers to num of cores {}".format(num_cores))

        self.train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.config.training_params.batch_size,
            num_workers=num_cores,
            shuffle=True,
            pin_memory=self.config.training_params.pin_memory,
            generator=g,
            worker_init_fn=lambda worker_id: np.random.seed(15 + worker_id),
        )
        self.valid_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=self.config.training_params.test_batch_size,
            shuffle=False,
            num_workers=num_cores,
            pin_memory=self.config.training_params.pin_memory,
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=self.config.training_params.test_batch_size,
            shuffle=False,
            num_workers=num_cores,
            pin_memory=self.config.training_params.pin_memory,
        )

    def _get_datasets(self):

        annotation_folder = os.path.join(self.config.dataset.data_roots, "ucfTrainTestlist")

        train_dataset = UCF_Dataset(
            config=self.config,
            mode="train",
            annotation_file=os.path.join(
                annotation_folder,
                "trainlist0{}.txt".format(int(self.config.dataset.fold)),
            ),
        )
        val_dataset = UCF_Dataset(
            config=self.config,
            mode="val",
            data=train_dataset.this_stored_data,
            annotation_file=os.path.join(
                annotation_folder,
                "trainlist0{}.txt".format(int(self.config.dataset.fold)),
            ),
        )
        del train_dataset.this_stored_data

        test_dataset = UCF_Dataset(
            config=self.config,
            mode="test",
            annotation_file=os.path.join(
                annotation_folder,
                "testlist0{}.txt".format(int(self.config.dataset.fold)),
            ),
        )

        return train_dataset, val_dataset, test_dataset
