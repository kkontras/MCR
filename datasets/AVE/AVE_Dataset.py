import logging
import os
import pickle
import random

import numpy as np
import torch
import torchaudio
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class AVEDataset(Dataset):

    def __init__(self, config, mode="train"):
        self.args = config
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode
        self.num_frame = config.dataset.get("num_frame", 4)

        self.data_root = self.args.dataset.data_roots

        self.visual_feature_path = r"{}".format(self.data_root)
        self.audio_feature_path = r"{}/Audio-1004-SE".format(self.data_root)

        self.train_txt = "./datasets/AVE/trainSet.txt"
        self.test_txt = "./datasets/AVE/testSet.txt"
        self.val_txt = "./datasets/AVE/valSet.txt"

        if mode == "train":
            txt_file = self.train_txt
        elif mode == "test":
            txt_file = self.test_txt
        else:
            txt_file = self.val_txt

        class_dict = {
            "Church bell": 0,
            "Male speech, man speaking": 1,
            "Bark": 2,
            "Fixed-wing aircraft, airplane": 3,
            "Race car, auto racing": 4,
            "Female speech, woman speaking": 5,
            "Helicopter": 6,
            "Violin, fiddle": 7,
            "Flute": 8,
            "Ukulele": 9,
            "Frying (food)": 10,
            "Truck": 11,
            "Shofar": 12,
            "Motorcycle": 13,
            "Acoustic guitar": 14,
            "Train horn": 15,
            "Clock": 16,
            "Banjo": 17,
            "Goat": 18,
            "Baby cry, infant cry": 19,
            "Bus": 20,
            "Chainsaw": 21,
            "Cat": 22,
            "Horse": 23,
            "Toilet flush": 24,
            "Rodents, rats, mice": 25,
            "Accordion": 26,
            "Mandolin": 27,
        }

        with open(txt_file, "r") as f2:
            files = f2.readlines()
            for item in files:
                item = item.split("&")
                audio_path = os.path.join(self.audio_feature_path, item[1] + ".pkl")
                visual_path = os.path.join(
                    self.visual_feature_path,
                    "Image-{:02d}-FPS-SE".format(self.args.dataset.fps),
                    item[1],
                )

                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    if audio_path not in self.audio:
                        self.image.append(visual_path)
                        self.audio.append(audio_path)
                        self.label.append(class_dict[item[0]])
                else:
                    print("Audio or visual path does not exist: ", audio_path, visual_path)
                    continue

    def __len__(self):
        return len(self.image)

    def _get_images(self, idx):
        if self.mode == "train":
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    # transforms.Resize(size=(224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        # Visual
        image_samples = os.listdir(self.image[idx])
        image_samples.sort()

        images = torch.zeros((self.args.dataset.num_frame, 3, 224, 224))
        for i in range(self.args.dataset.num_frame):

            img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert("RGB")
            img = transform(img)
            images[i] = img

        images = torch.permute(images, (1, 0, 2, 3))

        return images

    def _get_audio(self, idx):
        # audio
        spectrogram = pickle.load(open(self.audio[idx], "rb"))

        return spectrogram

    def __getitem__(self, idx):

        images = self._get_images(idx)
        spectrogram = self._get_audio(idx)
        label = self.label[idx]

        if self.mode == "test":
            random_idx = random.randint(0, len(self.image) - 1)
            sh_images = self._get_images(random_idx)
            sh_spectrogram = self._get_audio(random_idx)

            return {
                "data": {
                    0: spectrogram,
                    1: images,
                    "0_random_indistr": sh_spectrogram,
                    "1_random_indistr": sh_images,
                },
                "label": label,
            }

        return {"data": {0: spectrogram, 1: images}, "label": label}


class AVE_Dataloader:

    def __init__(self, config):
        """
        :param config:
        """
        self.config = config

        dataset_train, dataset_val, dataset_test, dataset_total = self._get_datasets()

        g = torch.Generator()
        g.manual_seed(0)

        num_cores = len(os.sched_getaffinity(0)) - 1

        print("Available cores {}".format(len(os.sched_getaffinity(0))))
        print("We are changing dataloader workers to num of cores {}".format(num_cores))

        logging.info("Train {}, Val {}, Test {}".format(len(dataset_train), len(dataset_val), len(dataset_test)))

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

        self.total_loader = torch.utils.data.DataLoader(
            dataset_total,
            batch_size=self.config.training_params.test_batch_size,
            shuffle=False,
            num_workers=num_cores,
            pin_memory=self.config.training_params.pin_memory,
        )

    def _get_datasets(self):

        train_dataset = AVEDataset(config=self.config, mode="train")
        valid_dataset = AVEDataset(config=self.config, mode="val")
        test_dataset = AVEDataset(config=self.config, mode="test")
        total_dataset = AVEDataset(config=self.config, mode="total")

        return train_dataset, valid_dataset, test_dataset, total_dataset


class AVEDataset_vit(Dataset):
    logger = logging.getLogger("AVE Dataset")
    logger.setLevel(logging.INFO)

    def __init__(self, config, mode="train"):
        self.args = config
        self.image, self.audio, self.audio_wav, self.label = [], [], [], []
        self.mode = mode
        self.num_frame = config.dataset.get("num_frame", 4)
        self.norm_type = config.dataset.get("norm_type", False)
        self.data_root = self.args.dataset.data_roots

        self.visual_feature_path = r"{}".format(self.data_root)
        self.audio_feature_path = r"{}/Audio-1004-SE".format(self.data_root)
        self.audiowav_feature_path = r"{}/Audios".format(self.data_root)

        self.sampling_rate = config.dataset.get("sampling_rate", 16000)
        self.max_duration = config.dataset.get("max_duration", 10)
        self.return_data = config.dataset.get("return_data", {"video": True, "spectrogram": True, "audio": False})

        self.train_txt = "./datasets/AVE/trainSet.txt"
        self.test_txt = "./datasets/AVE/testSet.txt"
        self.val_txt = "./datasets/AVE/valSet.txt"

        if mode == "train":
            txt_file = self.train_txt
        elif mode == "test":
            txt_file = self.test_txt
        else:
            txt_file = self.val_txt

        class_dict = {
            "Church bell": 0,
            "Male speech, man speaking": 1,
            "Bark": 2,
            "Fixed-wing aircraft, airplane": 3,
            "Race car, auto racing": 4,
            "Female speech, woman speaking": 5,
            "Helicopter": 6,
            "Violin, fiddle": 7,
            "Flute": 8,
            "Ukulele": 9,
            "Frying (food)": 10,
            "Truck": 11,
            "Shofar": 12,
            "Motorcycle": 13,
            "Acoustic guitar": 14,
            "Train horn": 15,
            "Clock": 16,
            "Banjo": 17,
            "Goat": 18,
            "Baby cry, infant cry": 19,
            "Bus": 20,
            "Chainsaw": 21,
            "Cat": 22,
            "Horse": 23,
            "Toilet flush": 24,
            "Rodents, rats, mice": 25,
            "Accordion": 26,
            "Mandolin": 27,
        }

        with open(txt_file, "r") as f2:
            files = f2.readlines()
            for item in files:
                item = item.split("&")
                audio_path = os.path.join(self.audio_feature_path, item[1] + ".pkl")
                audio_path_wav = os.path.join(self.audiowav_feature_path, item[1] + ".wav")
                visual_path = os.path.join(
                    self.visual_feature_path,
                    "Image-{:02d}-FPS-SE".format(self.args.dataset.fps),
                    item[1],
                )

                if self.return_data["spectrogram"]:
                    audio_flag = False
                    if os.path.exists(audio_path) and audio_path not in self.audio:
                        audio_flag = True
                else:
                    audio_flag = True

                if self.return_data["audio"]:
                    audio_wav_flag = False
                    if os.path.exists(audio_path_wav) and audio_path_wav not in self.audio_wav:
                        audio_wav_flag = True
                else:
                    audio_wav_flag = True

                if self.return_data["video"]:
                    video_flag = False
                    if os.path.exists(visual_path) and visual_path not in self.image:
                        video_flag = True
                else:
                    video_flag = True

                if video_flag and audio_flag and audio_wav_flag:
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    self.audio_wav.append(audio_path_wav)
                    self.label.append(class_dict[item[0]])
                else:
                    continue

        if self.args.dataset.get("norm_wav_path", False):
            self.wav_norm = pickle.loads(open(self.args.dataset.norm_wav_path, "rb").read())
        else:
            if mode == "train":
                self.get_wav_normalizer()
                raise Exception("Please run again the code, with config dataset.norm_wav_path equal to {}".format("./datasets/AVE/wav_norm.pkl"))

        # self.make_barplots(class_dict)

    def make_barplots(self, class_dict):
        num_images = []
        for idx in range(len(self.image)):
            num_images.append(len(os.listdir(self.image[idx])))
        num_images = np.array(num_images)

        import matplotlib.pyplot as plt

        num, count = np.unique(num_images, return_counts=True)
        plt.bar(num, count)
        plt.title(self.mode)
        plt.show()

        num, count = np.unique(self.label, return_counts=True)
        plt.bar(class_dict.keys(), count)
        plt.xticks(range(len(num)), class_dict.keys(), rotation="vertical")
        plt.title(self.mode)
        plt.tight_layout()
        plt.show()

    def __len__(self):
        if self.return_data["video"]:
            return len(self.image)
        elif self.return_data["audio"]:
            return len(self.audio)
        elif self.return_data["spectrogram"]:
            return len(self.audio_wav)

    def _get_images(self, idx):

        if not self.return_data["video"]:
            return False

        if self.mode == "train":
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    # transforms.Resize(size=(224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        # Visual
        image_samples = os.listdir(self.image[idx])
        image_samples.sort()

        images = []
        for i in range(self.args.dataset.num_frame):
            if i < len(image_samples):
                img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert("RGB")
                img = transform(img)
                images.append(img)

        images = torch.stack(images)

        return images

    def get_wav_normalizer(self):
        count = 0
        wav_sum = 0
        wav_sqsum = 0

        max_duration = 10

        for cur_wav in tqdm(self.audio_wav):
            audio, fps = torchaudio.load(cur_wav)

            audio = torchaudio.functional.resample(audio, fps, self.sampling_rate)

            audio = audio[0][: self.sampling_rate * max_duration]

            wav_sum += torch.sum(audio)
            wav_sqsum += torch.sum(audio**2)
            count += len(audio)

        wav_mean = wav_sum / count
        wav_var = (wav_sqsum / count) - (wav_mean**2)
        wav_std = np.sqrt(wav_var)

        self.wav_norm = {"mean": wav_mean, "std": wav_std}
        open("./datasets/AVE/wav_norm.pkl", "wb").write(pickle.dumps(self.wav_norm))

        return wav_mean, wav_std

    def _get_audio(self, idx):

        if not self.return_data["audio"]:
            return False

        audio, fps = torchaudio.load(self.audio_wav[idx])
        audio = torchaudio.functional.resample(audio, fps, self.sampling_rate)
        audio = audio[0]

        return audio

    def _get_spectrogram(self, idx):

        if not self.return_data["spectrogram"]:
            return False
        # audio
        spectrogram = pickle.load(open(self.audio[idx], "rb"))

        if self.norm_type == "per_sample":
            # Normalize per sample
            mean = np.mean(spectrogram)
            std = np.std(spectrogram)
            spectrogram = np.divide(spectrogram - mean, std + 1e-9)
        elif self.norm_type == "per_freq":
            # Normalize per freq
            mean = np.array(self.norm_audio["per_req"]["mean"])
            std = np.array(self.norm_audio["per_req"]["std"])
            spectrogram = np.divide(spectrogram.T - mean, std + 1e-9).T
        elif self.norm_type == "total":
            # Normalize per freq
            mean = self.norm_audio["total"]["mean"]
            std = self.norm_audio["total"]["std"]
            spectrogram = np.divide(spectrogram - mean, std + 1e-9)
        spectrogram = torch.from_numpy(spectrogram)

        return spectrogram

    def __getitem__(self, idx):

        images = self._get_images(idx)
        spectrogram = self._get_spectrogram(idx)
        audio = self._get_audio(idx)
        label = self.label[idx]

        return {"data": {0: spectrogram, 1: images, 2: audio}, "label": label}


def collate_fn_padd(batch):
    """
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    """

    aggregated_batch = {}
    for key in batch[0].keys():
        aggregated_batch[key] = {}
        if type(batch[0][key]) is int:
            aggregated_batch[key] = torch.LongTensor([d[key] for d in batch])

    key = "data"
    subkey = 0  # Spectrogram
    aggregated_list = [d[key][subkey].unsqueeze(dim=0) for d in batch if d[key][subkey] is not False]
    if len(aggregated_list) > 0:
        aggregated_batch[key][subkey] = torch.cat(aggregated_list, dim=0)

    subkey = 1  # Video
    aggregated_list = [d[key][subkey].unsqueeze(dim=0) for d in batch if d[key][subkey] is not False]

    if len(aggregated_list) > 0:

        length_list = [d.shape[1] for d in aggregated_list]
        aggregated_batch[key][subkey] = torch.nn.utils.rnn.pad_sequence(aggregated_list, batch_first=True)
        video_attention_mask = torch.zeros((len(aggregated_list), max(length_list)))
        for data_idx, dur in enumerate(length_list):
            video_attention_mask[data_idx, :dur] = 1
        aggregated_batch[key][subkey] = torch.cat(aggregated_list, dim=0)
        aggregated_batch[key]["attention_mask_video"] = video_attention_mask

    subkey = 2  # Audio
    aggregated_list = [d[key][subkey] for d in batch if d[key][subkey] is not False]
    if len(aggregated_list) > 0:
        length_list = [len(d) for d in aggregated_list]
        aggregated_batch[key][subkey] = torch.nn.utils.rnn.pad_sequence(aggregated_list, batch_first=True)
        audio_attention_mask = torch.zeros((len(aggregated_list), max(length_list)))
        for data_idx, dur in enumerate(length_list):
            audio_attention_mask[data_idx, :dur] = 1
        aggregated_batch[key]["attention_mask_audio"] = audio_attention_mask

    return aggregated_batch


class AVE_Dataloader_vit:

    def __init__(self, config):
        """
        :param config:
        """
        self.config = config

        dataset_train, dataset_val, dataset_test, dataset_total = self._get_datasets()

        g = torch.Generator()
        g.manual_seed(0)
        num_cores = len(os.sched_getaffinity(0)) - 1
        # num_cores = 0
        print("Available cores {}".format(len(os.sched_getaffinity(0))))
        print("We are changing dataloader workers to num of cores {}".format(num_cores))

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        dataset_train.logger.info("Train {}, Val {}, Test {}".format(len(dataset_train), len(dataset_val), len(dataset_test)))

        self.train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.config.training_params.batch_size,
            num_workers=num_cores,
            shuffle=True,
            pin_memory=self.config.training_params.pin_memory,
            generator=g,
            collate_fn=collate_fn_padd,
            worker_init_fn=seed_worker,
        )
        self.valid_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=self.config.training_params.test_batch_size,
            shuffle=False,
            num_workers=num_cores,
            collate_fn=collate_fn_padd,
            pin_memory=self.config.training_params.pin_memory,
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=self.config.training_params.test_batch_size,
            shuffle=False,
            num_workers=num_cores,
            collate_fn=collate_fn_padd,
            pin_memory=self.config.training_params.pin_memory,
        )

    def _get_datasets(self):

        train_dataset = AVEDataset_vit(config=self.config, mode="train")
        valid_dataset = AVEDataset_vit(config=self.config, mode="val")
        test_dataset = AVEDataset_vit(config=self.config, mode="test")
        total_dataset = AVEDataset_vit(config=self.config, mode="total")

        return train_dataset, valid_dataset, test_dataset, total_dataset
