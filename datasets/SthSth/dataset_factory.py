from torch.utils.data import Dataset
from yacs.config import CfgNode as CN

from datasets.SthSth.audio_dataset import AudioDataset
from datasets.SthSth.layout_dataset import LayoutDataset
from datasets.SthSth.video_dataset import VideoDataset


class VideoFlow(Dataset):
    # FIXME: Hacky
    def __init__(self, cfg, train: bool = False):
        # Flow dataset
        flow_cfg = cfg.clone()
        flow_cfg.defrost()
        flow_cfg.DATASET_TYPE = "flow"
        flow_cfg.freeze()
        self.flow_dataset = VideoDataset(flow_cfg, train=train)
        # video dataset
        video_cfg = cfg.clone()
        video_cfg.defrost()
        video_cfg.DATASET_TYPE = "video"
        video_cfg.freeze()
        self.video_dataset = VideoDataset(video_cfg, train=train)

    def __len__(self):
        return self.flow_dataset.__len__()

    def __getitem__(self, idx: int):
        flow_dict = self.flow_dataset[idx]
        self.video_dataset.set_indices(flow_dict["indices"])
        self.video_dataset.set_existing_transforms(self.flow_dataset.enforced_transforms)
        video_dict = self.video_dataset[idx]
        # Gather from both dicts
        output = {}
        for c_dict in [flow_dict, video_dict]:
            for key in c_dict.keys():
                output[key] = c_dict[key]

        return output


class VideoAudio(Dataset):
    # FIXME: Hacky
    def __init__(self, cfg, train: bool = False):
        # Audio dataset
        audio_cfg = cfg.clone()
        audio_cfg.defrost()
        audio_cfg.DATASET_TYPE = "audio"
        audio_cfg.freeze()
        self.audio_dataset = AudioDataset(audio_cfg, train=train)
        # video dataset
        video_cfg = cfg.clone()
        video_cfg.defrost()
        video_cfg.DATASET_TYPE = "video"
        video_cfg.freeze()
        self.video_dataset = VideoDataset(video_cfg, train=train)

    def __len__(self):
        return self.audio_dataset.__len__()

    def __getitem__(self, idx: int):
        audio_dict = self.audio_dataset[idx]
        self.video_dataset.set_indices(audio_dict["indices"])
        video_dict = self.video_dataset[idx]
        # Gather from both dicts
        output = {}
        for c_dict in [audio_dict, video_dict]:
            for key in c_dict.keys():
                output[key] = c_dict[key]

        return output


class VideoFlowAudio(Dataset):
    # FIXME: Hacky
    def __init__(self, cfg, train: bool = False):
        # Audio dataset
        audio_cfg = cfg.clone()
        audio_cfg.defrost()
        audio_cfg.DATASET_TYPE = "audio"
        audio_cfg.freeze()
        self.audio_dataset = AudioDataset(audio_cfg, train=train)
        # Flow dataset
        flow_cfg = cfg.clone()
        flow_cfg.defrost()
        flow_cfg.DATASET_TYPE = "flow"
        flow_cfg.freeze()
        self.flow_dataset = VideoDataset(flow_cfg, train=train)
        # video dataset
        video_cfg = cfg.clone()
        video_cfg.defrost()
        video_cfg.DATASET_TYPE = "video"
        video_cfg.freeze()
        self.video_dataset = VideoDataset(video_cfg, train=train)

    def set_weighted(self):
        from copy import deepcopy

        copy_self = deepcopy(self)
        copy_self.audio_dataset = self.audio_dataset.set_weighted()
        copy_self.video_dataset = self.video_dataset.set_weighted()
        copy_self.flow_dataset = self.flow_dataset.set_weighted()

        return copy_self

    def __len__(self):
        return self.flow_dataset.__len__()

    def __getitem__(self, idx: int):
        audio_dict = self.audio_dataset[idx]
        self.flow_dataset.set_indices(audio_dict["indices"])
        flow_dict = self.flow_dataset[idx]
        self.video_dataset.set_indices(audio_dict["indices"])
        self.video_dataset.set_existing_transforms(self.flow_dataset.enforced_transforms)
        video_dict = self.video_dataset[idx]
        # Gather from both dicts
        output = {}
        for c_dict in [flow_dict, video_dict, audio_dict]:
            for key in c_dict.keys():
                output[key] = c_dict[key]

        return output


class VideoLayoutDataset(Dataset):
    # FIXME: Hacky
    def __init__(self, cfg, train: bool = False):
        self.layout_dataset = LayoutDataset(cfg, train=train)
        video_cfg = cfg.clone()
        video_cfg.defrost()
        video_cfg.DATASET_TYPE = "video"
        video_cfg.freeze()
        self.video_dataset = VideoDataset(video_cfg, train=train)

    def __len__(self):
        return self.layout_dataset.__len__()

    def __getitem__(self, idx: int):
        layout_dict = self.layout_dataset[idx]
        self.video_dataset.set_indices(layout_dict["indices"])
        video_dict = self.video_dataset[idx]
        # Gather from both dicts
        output = {}
        for c_dict in [layout_dict, video_dict]:
            for key in c_dict.keys():
                output[key] = c_dict[key]

        return output


class VideoLayoutFlow(Dataset):
    def __init__(self, config, train: bool = False):
        # FIXME: Hacky

        self.config = config

        if self.config.dataset.modalities.layout.activate:

            # lay_cfg = get_cfg_defaults()
            # lay_cfg.merge_from_file("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2023_data/SthSth/gorjan_experiments/unimodal_layout/config.yaml")
            lay_cfg = CN()
            lay_cfg.defrost()
            lay_cfg.DATASET_TYPE = self.config.dataset.modalities.layout.dataset_type
            lay_cfg.TRAIN_DATASET_PATH = self.config.dataset.modalities.layout.train_dataset_path
            lay_cfg.TRAIN_DATASET_NAME = self.config.dataset.modalities.layout.train_dataset_name
            lay_cfg.VAL_DATASET_NAME = self.config.dataset.modalities.layout.val_dataset_name
            lay_cfg.LABELS_PATH = self.config.dataset.modalities.layout.labels_path
            lay_cfg.VAL_DATASET_PATH = self.config.dataset.modalities.layout.val_dataset_path
            lay_cfg.MODEL_NAME = self.config.dataset.model_name
            lay_cfg.STRIDE = self.config.dataset.stride
            lay_cfg.NUM_TEST_CLIPS = self.config.dataset.num_test_clips
            lay_cfg.NUM_TEST_CROPS = self.config.dataset.num_test_crops
            lay_cfg.FRAME_UNIFORM = self.config.dataset.frame_uniform
            lay_cfg.NUM_FRAMES = self.config.dataset.num_frames
            lay_cfg.AUGMENTATIONS = CN()
            lay_cfg.AUGMENTATIONS.LAYOUT = self.config.dataset.augmentations.layout
            lay_cfg.freeze()

            self.layout_dataset = LayoutDataset(lay_cfg, train=train)

        if self.config.dataset.modalities.flow.activate:

            # Flow dataset
            flow_cfg = CN()
            flow_cfg.defrost()
            flow_cfg.DATASET_TYPE = self.config.dataset.modalities.flow.dataset_type
            flow_cfg.TRAIN_DATASET_PATH = self.config.dataset.modalities.flow.train_dataset_path
            flow_cfg.TRAIN_DATASET_NAME = self.config.dataset.modalities.flow.train_dataset_name
            flow_cfg.VAL_DATASET_NAME = self.config.dataset.modalities.flow.val_dataset_name
            flow_cfg.LABELS_PATH = self.config.dataset.modalities.flow.labels_path
            flow_cfg.VAL_DATASET_PATH = self.config.dataset.modalities.flow.val_dataset_path
            flow_cfg.FLOW_PATH = self.config.dataset.modalities.flow.flow_path
            flow_cfg.MODEL_NAME = self.config.dataset.model_name
            flow_cfg.STRIDE = self.config.dataset.stride
            flow_cfg.NUM_TEST_CLIPS = self.config.dataset.num_test_clips
            flow_cfg.NUM_TEST_CROPS = self.config.dataset.num_test_crops
            flow_cfg.FRAME_UNIFORM = self.config.dataset.frame_uniform
            flow_cfg.NUM_FRAMES = self.config.dataset.num_frames
            flow_cfg.AUGMENTATIONS = CN()
            flow_cfg.AUGMENTATIONS.FLOW = self.config.dataset.augmentations.flow
            flow_cfg.freeze()
            self.flow_dataset = VideoDataset(flow_cfg, train=train)

        if self.config.dataset.modalities.video.activate:

            # video dataset

            video_cfg = CN()
            video_cfg.defrost()
            video_cfg.DATASET_TYPE = self.config.dataset.modalities.video.dataset_type
            video_cfg.TRAIN_DATASET_PATH = self.config.dataset.modalities.video.train_dataset_path
            video_cfg.TRAIN_DATASET_NAME = self.config.dataset.modalities.video.train_dataset_name
            video_cfg.VAL_DATASET_NAME = self.config.dataset.modalities.video.val_dataset_name
            video_cfg.LABELS_PATH = self.config.dataset.modalities.video.labels_path
            video_cfg.VAL_DATASET_PATH = self.config.dataset.modalities.video.val_dataset_path
            video_cfg.VIDEOS_PATH = self.config.dataset.modalities.video.videos_path
            video_cfg.MODEL_NAME = self.config.dataset.model_name
            video_cfg.STRIDE = self.config.dataset.stride
            video_cfg.NUM_TEST_CLIPS = self.config.dataset.num_test_clips
            video_cfg.NUM_TEST_CROPS = self.config.dataset.num_test_crops
            video_cfg.FRAME_UNIFORM = self.config.dataset.frame_uniform
            video_cfg.NUM_FRAMES = self.config.dataset.num_frames
            video_cfg.AUGMENTATIONS = CN()
            video_cfg.AUGMENTATIONS.VIDEO = self.config.dataset.augmentations.video
            video_cfg.freeze()

            self.video_dataset = VideoDataset(video_cfg, train=train)

    def __len__(self):

        if hasattr(self, "layout_dataset"):
            total_len = len(self.layout_dataset)
        elif hasattr(self, "flow_dataset"):
            total_len = len(self.flow_dataset)
        elif hasattr(self, "video_dataset"):
            total_len = len(self.video_dataset)
        return total_len

    def __getitem__(self, idx: int):

        output = {"data": {}}
        common_indices = None
        if self.config.dataset.modalities.flow.activate:
            flow_dict = self.flow_dataset[idx]
            common_indices = flow_dict["indices"]
            output["data"][2] = flow_dict["flow"].squeeze(dim=0).permute(1, 0, 2, 3)
            output["label"] = flow_dict["labels"]["ACTION"]

        if self.config.dataset.modalities.video.activate:
            if common_indices is not None:
                self.video_dataset.set_indices(common_indices)
                self.video_dataset.set_existing_transforms(self.flow_dataset.enforced_transforms)
            video_dict = self.video_dataset[idx]
            common_indices = video_dict["indices"]
            output["data"][1] = video_dict["video"].squeeze().permute(1, 0, 2, 3)
            output["label"] = video_dict["labels"]["ACTION"]

        if self.config.dataset.modalities.layout.activate:
            if common_indices is not None:
                self.layout_dataset.set_indices(common_indices)
            layout_dict = self.layout_dataset.__getitem__(idx)
            output["data"][0] = layout_dict["bboxes"]
            output["data"]["bboxes"] = layout_dict["bboxes"]
            output["data"]["scores"] = layout_dict["scores"]
            output["data"]["sides"] = layout_dict["sides"]
            output["data"]["states"] = layout_dict["states"]
            output["data"]["class_labels"] = layout_dict["class_labels"]
            output["data"]["src_key_padding_mask_boxes"] = layout_dict["src_key_padding_mask_boxes"]
            # output["data"]["labels"] = layout_dict["labels"]
            output["label"] = layout_dict["labels"]["ACTION"]
            output["indices"] = layout_dict["indices"]
            output["start_frame"] = layout_dict["start_frame"]

        return output


dataset_factory = {
    "video": VideoDataset,
    "flow": VideoDataset,
    "depth": VideoDataset,
    "layout": LayoutDataset,
    "video_layout": VideoLayoutDataset,
    "video_flow": VideoFlow,
    "video_layout_flow": VideoLayoutFlow,
    "video_flow_audio": VideoFlowAudio,
    "audio": AudioDataset,
    "video_audio": VideoAudio,
}


import os

import numpy as np
import torch


class SthSth_VideoLayoutFlow:

    def __init__(self, config):
        """
        :param config:
        """
        self.config = config

        dataset_train, dataset_val = self._get_datasets()

        g = torch.Generator()
        g.manual_seed(0)

        # os.system("taskset -c -p 0-95 %d" % os.getpid())

        num_cores = len(os.sched_getaffinity(0)) - 1
        # num_cores = 12

        print("Available cores {}".format(len(os.sched_getaffinity(0))))
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

        train_dataset = VideoLayoutFlow(config=self.config, train=True)
        valid_dataset = VideoLayoutFlow(config=self.config, train=False)

        return train_dataset, valid_dataset


# from utils_setup import get_cfg_defaults
# #
# cfg = get_cfg_defaults()
# cfg.merge_from_file("/users/sista/kkontras/Documents/Balance/datasets/SthSth/config.yaml")
#
# cfg.BACKBONE_MODEL_PATH = None
# # Freeze the config
# cfg.freeze()
# a = dataset_factory["video_layout_flow"](cfg)
# print(len(a.video_dataset))
# print(len(a.layout_dataset))
# print(len(a.flow_dataset))
# b = a.__getitem__(1)
# print(b.keys())
# print(b["flow"].shape)
# print(b["video"].shape)
# print(b["bboxes"].shape)
#
# import matplotlib.pyplot as plt
#
# fig, axes = plt.subplots(4, 4, figsize=(10, 10))
#
# # Flatten the 2D array of axes into a 1D array
# axes = axes.flatten()
#
# # Loop through the images and plot them in each subplot
# for i in range(len(b["flow"][0])):
#     axes[i].imshow(b["flow"][0][i].permute(1,2,0))  # Assuming grayscale images, adjust cmap as needed
#     axes[i].axis('off')  # Turn off axis labels for cleaner display
#
# # Adjust layout for better spacing
# plt.tight_layout()
#
# # Show the plot
# plt.show()
#
#
# fig, axes = plt.subplots(4, 4, figsize=(10, 10))
# axes = axes.flatten()
# for i in range(len(b["video"][0])):
#     axes[i].imshow(b["video"][0][i].permute(1,2,0))  # Assuming grayscale images, adjust cmap as needed
#     axes[i].axis('off')  # Turn off axis labels for cleaner display
# plt.tight_layout()
# plt.show()
