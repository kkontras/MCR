import copy
import csv
import os
import pickle
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pdb
import random
import json
from sklearn.model_selection import train_test_split
import multiprocessing

# from transformers import VivitImageProcessor
import einops
import torchaudio
from tqdm import tqdm
from collections import defaultdict
import logging
import librosa


class CremadDataset(Dataset):

    def __init__(self, config, fps=1, mode="train"):
        self.image = []
        # self.face_image = []
        self.audio = []
        self.faces = []
        self.label = []
        self.mode = mode
        self.fps = fps
        self.config = config
        self.num_frame = self.config.dataset.get("num_frame", 3)
        data_split = self.config.dataset.get("data_split", {"a": 0})
        fold = data_split.get("fold", 0)
        self.norm_type = self.config.dataset.get("norm_type", False)
        self.config = self.config
        self.sampling_rate = self.config.dataset.sampling_rate

        # self.image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")

        self.return_data = config.dataset.get("return_data", {"video": True, "spectrogram": True, "audio": False, "face": False, "face_image": False})

        class_dict = {"NEU": 0, "HAP": 1, "SAD": 2, "FEA": 3, "DIS": 4, "ANG": 5}

        self.visual_feature_path = self.config.dataset.data_roots
        self.audio_feature_path = os.path.join(self.config.dataset.data_roots, "AudioWAV")
        self.face_feature_path = os.path.join(self.config.dataset.data_roots, "Face_features")
        # self.face_image_path = os.path.join(self.config.dataset.data_roots, "Face_features_images")

        if data_split.get("method", "inclusive") == "inclusive":
            self.split_inclusive(mode, class_dict)
        elif data_split.get("method", "inclusive") == "non_inclusive":
            self.split_noninclusive(fold, mode, class_dict)
        else:
            raise ValueError(
                "config.dataset.data_split should be either 'inclusive' or 'non_inclusive', then {} is not an option".format(
                    self.config.dataset.get("data_split", "inclusive")
                )
            )

        if self.config.dataset.get("norm_wav_path", None) and os.path.exists(self.config.dataset.get("norm_wav_path", None)):
            self.wav_norm = pickle.loads(open(self.config.dataset.norm_wav_path, "rb").read())
            logging.info("Loaded wav norm from {}".format(self.config.dataset.norm_wav_path))
            logging.info("Norm values are {}".format(self.wav_norm))
        else:
            if mode == "train":
                self.get_wav_normalizer()
                save_dir = (
                    self.config.dataset.get("norm_wav_path", None) if self.config.dataset.get("norm_wav_path", None) is not None else "./datasets/CREMAD/wav_norm.pkl"
                )
                logging.warning("Saving wav norm to {}".format(save_dir))

        if self.config.dataset.get("norm", True):
            if self.config.dataset.get("norm_face_path", None) and os.path.exists(self.config.dataset.get("norm_face_path", None)):
                self.face_norm = pickle.loads(open(self.config.dataset.norm_face_path, "rb").read())
                logging.info("Loaded face norm from {}".format(self.config.dataset.norm_face_path))
                logging.info("Norm values are {}".format(self.face_norm))
            else:
                if mode == "train":
                    self.get_face_normalizer()
                    save_dir = (
                        self.config.dataset.get("norm_face_path", None)
                        if self.config.dataset.get("norm_face_path", None) is not None
                        else "./datasets/CREMAD/face_norm.pkl"
                    )
                    logging.warning("Saving face norm to {}".format(save_dir))

    def make_barplots(self, class_dict):
        num_images = []
        for idx in range(len(self.image)):
            num_images.append(len(os.listdir(str(self.image[idx]))))
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

    def split_inclusive(self, mode, class_dict):

        # with open('./datasets/CREMAD/normalization_audio_OGM.pkl', "r") as json_file:
        #     self.norm_audio = json.load(json_file)
        #     self.norm_audio = json.loads(self.norm_audio)

        self.norm_audio = {"total": {"mean": -7.1276217, "std": 5.116028}}

        (
            self.train_image,
            self.train_face_image,
            self.train_audio,
            self.train_faces,
            self.train_label,
            self.train_item,
        ) = ([], [], [], [], [], [])
        self.test_image, self.test_face_image, self.test_audio, self.test_faces, self.test_label, self.test_item = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        self.train_csv = "./datasets/CREMAD/train.csv"
        self.test_csv = "./datasets/CREMAD/test.csv"
        with open(self.train_csv, encoding="UTF-8-sig") as f2:
            csv_reader = csv.reader(f2)
            for item in csv_reader:
                audio_path = os.path.join(self.audio_feature_path, item[0] + ".wav")
                visual_path = os.path.join(self.visual_feature_path, "Image-01-FPS", item[0])
                face_path = os.path.join(self.face_feature_path, item[0] + ".npy")
                # face_image_path = os.path.join(self.face_image_path,  item[0] + '.npy')
                if os.path.exists(audio_path) and os.path.exists(visual_path) and os.path.exists(face_path):
                    self.train_item.append(item[0])
                    self.train_image.append(visual_path)
                    # self.train_face_image.append(face_image_path)
                    self.train_audio.append(audio_path)
                    self.train_faces.append(audio_path)
                    self.train_label.append(class_dict[item[1]])
                else:
                    continue
        with open(self.test_csv, encoding="UTF-8-sig") as f2:
            csv_reader = csv.reader(f2)
            for item in csv_reader:
                audio_path = os.path.join(self.audio_feature_path, item[0] + ".wav")
                visual_path = os.path.join(self.visual_feature_path, "Image-01-FPS", item[0])
                face_path = os.path.join(self.face_feature_path, item[0] + ".npy")
                # face_image_path = os.path.join(self.face_image_path,  item[0] + '.npy')
                if os.path.exists(audio_path) and os.path.exists(visual_path) and os.path.exists(face_path):
                    self.test_item.append(item[0])
                    self.test_image.append(visual_path)
                    self.test_audio.append(audio_path)
                    self.test_faces.append(face_path)
                    # self.test_face_image.append(face_image_path)
                    self.test_label.append(class_dict[item[1]])
                else:
                    continue

        if self.split_mode == "stratified_split":

            self.total_item = self.train_item + self.test_item
            self.total_image = self.train_image + self.test_image
            self.total_audio = self.train_audio + self.test_audio
            self.total_faces = self.train_faces + self.test_faces
            # self.total_face_image = self.train_face_image + self.test_face_image
            self.total_label = self.train_label + self.test_label
            X_trainval, X_test, y_trainval, y_test = train_test_split(
                # np.array([self.total_item, self.total_image, self.total_audio, self.total_faces, self.total_face_image]).T,
                np.array([self.total_item, self.total_image, self.total_audio, self.total_faces]).T,
                np.array(self.total_label),
                test_size=self.config.dataset.get("val_split_rate", 0.1),
                random_state=self.config.training_params.seed,
                stratify=np.array(self.total_label),
            )

            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval,
                y_trainval,
                test_size=self.config.dataset.get("val_split_rate", 0.1),
                random_state=self.config.training_params.seed,
                stratify=y_trainval,
            )

            if mode == "test":
                X = X_test
                y = y_test
            elif mode == "train":
                X = X_train
                y = y_train
            elif mode == "val":
                X = X_val
                y = y_val

            self.item = X[:, 0]
            self.image = X[:, 1]
            self.audio = X[:, 2]
            self.faces = X[:, 3]
            self.label = y

        elif self.split_mode == "val_stratified_split":

            if mode == "test":

                self.item = self.test_item[: int(len(self.test_item))]
                self.image = self.test_image[: int(len(self.test_image))]
                self.audio = self.test_audio[: int(len(self.test_audio))]
                self.faces = self.test_faces[: int(len(self.test_faces))]
                self.label = self.test_label[: int(len(self.test_label))]

            elif mode == "val" or mode == "train":

                split_data = np.array([self.train_item, self.train_image, self.train_audio, self.train_faces]).T
                X_train, X_val, y_train, y_val = train_test_split(
                    split_data,
                    np.array(self.train_label),
                    test_size=self.config.dataset.get("val_split_rate", 0.1),
                    random_state=self.config.training_params.seed,
                    stratify=np.array(self.train_label),
                )

                if mode == "train":
                    X = X_train
                    y = y_train
                elif mode == "val":
                    X = X_val
                    y = y_val

                self.item = X[:, 0]
                self.image = X[:, 1]
                self.audio = X[:, 2]
                self.faces = X[:, 3]
                self.label = y

        elif self.split_mode == "90_percent":
            if mode == "train":
                self.item = self.train_item[: int(len(self.train_item) * 0.9)]
                self.image = self.train_image[: int(len(self.train_image) * 0.9)]
                self.audio = self.train_audio[: int(len(self.train_audio) * 0.9)]
                self.faces = self.train_faces[: int(len(self.train_faces) * 0.9)]
                self.label = self.train_label[: int(len(self.train_label) * 0.9)]
            elif mode == "test":
                self.item = self.test_item[: int(len(self.test_item) * 0.9)]
                self.image = self.test_image[: int(len(self.test_image) * 0.9)]
                self.audio = self.test_audio[: int(len(self.test_audio) * 0.9)]
                self.faces = self.test_faces[: int(len(self.test_faces) * 0.9)]
                self.label = self.test_label[: int(len(self.test_label) * 0.9)]
            elif mode == "val":
                self.item = self.train_item[int(len(self.train_item) * 0.9) :]
                for k in self.test_item[int(len(self.test_item) * 0.9) :]:
                    self.item.append(k)
                self.image = self.train_image[int(len(self.train_image) * 0.9) :]
                for k in self.test_image[int(len(self.test_image) * 0.9) :]:
                    self.image.append(k)

                self.audio = self.train_audio[int(len(self.train_audio) * 0.9) :]
                for k in self.test_audio[int(len(self.test_audio) * 0.9) :]:
                    self.audio.append(k)

                self.faces = self.train_faces[int(len(self.train_faces) * 0.9) :]
                for k in self.test_faces[int(len(self.test_faces) * 0.9) :]:
                    self.faces.append(k)

                self.label = self.train_label[int(len(self.train_label) * 0.9) :]
                for k in self.test_label[int(len(self.test_label) * 0.9) :]:
                    self.label.append(k)

        elif self.split_mode == "90_percent_val":
            if mode == "train":
                self.item = self.train_item[: int(len(self.train_item) * 0.9)]
                self.image = self.train_image[: int(len(self.train_image) * 0.9)]
                self.audio = self.train_audio[: int(len(self.train_audio) * 0.9)]
                self.faces = self.train_faces[: int(len(self.train_faces) * 0.9)]
                self.label = self.train_label[: int(len(self.train_label) * 0.9)]
            elif mode == "test":
                self.item = self.test_item
                self.image = self.test_image
                self.audio = self.test_audio
                self.faces = self.test_faces
                self.label = self.test_label
            elif mode == "val":
                self.item = self.train_item[int(len(self.train_item) * 0.9) :]
                self.image = self.train_image[int(len(self.train_image) * 0.9) :]
                self.audio = self.train_audio[int(len(self.train_audio) * 0.9) :]
                self.faces = self.train_faces[int(len(self.train_faces) * 0.9) :]
                self.label = self.train_label[int(len(self.train_label) * 0.9) :]

    def split_noninclusive(self, fold, mode, class_dict):

        with open("./datasets/CREMAD/normalization_audio.pkl", "r") as json_file:
            self.norm_audio = json.load(json_file)

        with open("./datasets/CREMAD/data_splits_VALV.pkl", "r") as json_file:
            train_val_test_splits = json.load(json_file)

        self.label_from_name = []
        for i in train_val_test_splits[str(fold + 1)][mode]:
            name = i.split("-")[0]
            # label = int(i.split("-")[1])
            label_from_name = class_dict[i.split("-")[0].split("_")[2]]
            ap = os.path.join(self.audio_feature_path, name)
            vp = os.path.join(self.visual_feature_path, "Image-{:02d}-FPS".format(self.fps), name.split(".")[0])
            fp = os.path.join(self.face_feature_path, name.replace(".wav", ".npy"))
            # fi = os.path.join(self.face_image_path, name.replace(".wav", ".npy"))

            if os.path.exists(ap) and os.path.exists(vp) and os.path.exists(fp):
                self.image.append(vp)
                self.audio.append(ap)
                self.faces.append(fp)
                self.label.append(label_from_name)
            else:
                print("This path does not exist {} or {} or {}".format(ap, vp, fp))
                continue

    def get_wav_normalizer(self):

        count = 0
        wav_sum = 0
        wav_sqsum = 0

        max_duration = 0  # seconds

        for cur_wav in tqdm(self.audio):
            audio, fps = torchaudio.load(cur_wav)
            audio = torchaudio.functional.resample(audio, fps, self.sampling_rate)
            audio = audio[0]
            if audio.shape[0] > max_duration * self.sampling_rate:
                max_duration = audio.shape[0] / self.sampling_rate
            wav_sum += torch.sum(audio)
            wav_sqsum += torch.sum(audio**2)
            count += len(audio)

        wav_mean = wav_sum / count
        wav_var = (wav_sqsum / count) - (wav_mean**2)
        wav_std = np.sqrt(wav_var)

        self.wav_norm = {"mean": wav_mean, "std": wav_std, "max_duration": max_duration}

        if self.config.dataset.get("norm_wav_path", None) is not None:
            open(self.config.dataset.norm_wav_path, "wb").write(pickle.dumps(self.wav_norm))
        else:
            open("./datasets/CREMAD/wav_norm.pkl", "wb").write(pickle.dumps(self.wav_norm))

        return wav_mean, wav_std

    def get_face_normalizer(self):

        count = 0
        vid_sum = 0
        vid_sqsum = 0
        max_faces = 0
        for cur_vid in tqdm(self.faces):
            feats = np.load(cur_vid)  # np.transpose(np.load(vid_loc + '.npy'))
            vid_sum += np.sum(feats, axis=0)
            vid_sqsum += np.sum(feats**2, axis=0)
            count += np.shape(feats)[0]
            if np.shape(feats)[0] > max_faces:
                max_faces = np.shape(feats)[0]

        vid_mean = vid_sum / count
        vid_var = (vid_sqsum / count) - (vid_mean**2)
        vid_std = np.sqrt(vid_var)

        self.face_norm = {"mean": vid_mean, "std": vid_std, "max_faces": max_faces}
        print("max faces: ", max_faces)

        if self.config.dataset.get("norm_face_path", None) is not None:
            open(self.config.dataset.norm_face_path, "wb").write(pickle.dumps(self.face_norm))
        else:
            open("./datasets/CREMAD/norm_face_path.pkl", "wb").write(pickle.dumps(self.face_norm))

        return vid_mean, vid_std

    def __len__(self):
        return len(self.image)

    def _get_images(self, idx):

        if not self.return_data["video"]:
            return False

        if self.mode == "train":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomResizedCrop(224, antialias=True),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(size=(224, 224), antialias=True),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        image_samples = os.listdir(str(self.image[idx]))
        image_samples.sort()

        images = torch.zeros((self.num_frame, 3, 224, 224))
        for i in range(len(image_samples)):
            img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert("RGB")
            img = transform(img)
            images[i] = img
            if i == self.num_frame - 1:
                break

        images = torch.permute(images, (1, 0, 2, 3))
        return images

    def _get_face(self, idx):

        if not self.return_data["face"]:
            return False

        face_features = np.load(self.faces[idx])
        face_features = torch.from_numpy(face_features)

        # if len(face_features.shape) > 2:
        #     os.remove(self.faces[idx])
        #     print("Deleted: ", self.faces[idx])

        face_features = (face_features - self.face_norm["mean"]) / self.face_norm["std"]

        return face_features

    def _get_face_image(self, idx):

        if not self.return_data.get("face_image", False):
            return False

        face_features = np.load(self.face_image[idx])
        face_features = torch.from_numpy(face_features)

        return face_features

    def _get_audio(self, idx):

        if not self.return_data["audio"]:
            return False

        audio, fps = torchaudio.load(self.audio[idx])
        audio = torchaudio.functional.resample(audio, fps, self.sampling_rate)
        # max_duration = 10 #seconds
        #
        # audio = audio[0][:int(self.sampling_rate*max_duration)]
        # audio = audio[0][:int(self.sampling_rate*3])]
        # audio = audio[0][:int(self.sampling_rate*self.wav_norm["max_duration"])]
        audio = audio[0]

        if hasattr(self, "wav_norm"):
            audio = (audio - self.wav_norm["mean"]) / self.wav_norm["std"]

        return audio

    def _get_spectrogram(self, idx, audio):

        if not self.return_data["spectrogram"]:
            return False

        if audio is False:
            samples, rate = librosa.load(self.audio[idx], sr=self.sampling_rate)
            resamples = np.tile(samples, 3)[: self.sampling_rate * 3]
            resamples[resamples > 1.0] = 1.0
            resamples[resamples < -1.0] = -1.0

        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        spectrogram = torch.from_numpy(spectrogram)

        #     audio, fps = torchaudio.load(self.audio[idx])
        #     audio = torchaudio.functional.resample(audio, fps, self.sampling_rate)
        #     max_duration = 3 #seconds
        #     audio = audio[0][:self.sampling_rate*max_duration]
        #
        #     # audio = (audio - self.wav_norm["mean"]) / self.wav_norm["std"]
        #
        #     #pad if less than 3 seconds
        #     if audio.shape[0] < self.sampling_rate*max_duration:
        #         audio = torch.cat((audio, torch.zeros(self.sampling_rate*max_duration - audio.shape[0])))
        #
        #
        # spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=353)(audio)
        # spectrogram = torch.log(torch.abs(spectrogram) + 1e-7)

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

        return spectrogram

    def __getitem__(self, idx):

        images = self._get_images(idx)
        audio = self._get_audio(idx)
        face_features = self._get_face(idx)
        # face_image = self._get_face_image(idx)
        spectrogram = self._get_spectrogram(idx, audio)
        label = self.label[idx]

        # if self.mode=="test":
        #     random_idx = random.randint(0, len(self.image)-1)
        #     sh_images = self._get_images(random_idx)
        #     sh_spectrogram = self._get_audio(random_idx)
        #
        #     return {"data": {0:spectrogram, 1:images, "0_random_indistr": sh_spectrogram, "1_random_indistr": sh_images}, "label": label}
        # print idx with all output shape
        # print("idx: {}, spectrogram: {}, images: {}, audio: {}".format(idx, spectrogram.shape, images.shape, audio.shape))

        # return {"data":{0:spectrogram, 1:images, 2:audio, 3:face_features, 4:face_image},"label": label, "idx": idx}
        return {"data": {0: spectrogram, 1: images, 2: audio, 3: face_features}, "label": label, "idx": idx}


def collate_fn_padd(batch):

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
        aggregated_batch[key][subkey] = torch.cat(aggregated_list, dim=0)

    subkey = 2  # Audio
    aggregated_list = [d[key][subkey] for d in batch if d[key][subkey] is not False]
    if len(aggregated_list) > 0:
        length_list = [len(d) for d in aggregated_list]
        aggregated_batch[key][subkey] = torch.nn.utils.rnn.pad_sequence(aggregated_list, batch_first=True)
        audio_attention_mask = torch.zeros((len(aggregated_list), max(length_list)))
        for data_idx, dur in enumerate(length_list):
            audio_attention_mask[data_idx, :dur] = 1
        aggregated_batch[key]["attention_mask_audio"] = audio_attention_mask

    subkey = 3  # Face
    aggregated_list = [d[key][subkey] for d in batch if d[key][subkey] is not False]
    if len(aggregated_list) > 0:
        length_list = [len(d) for d in aggregated_list]
        max_length = max(length_list)
        if max_length > 150:
            # print("Here length was {}".format(length_list))
            max_length = 150
            aggregated_list = [i[:max_length] for i in aggregated_list]
        aggregated_batch[key][subkey] = torch.nn.utils.rnn.pad_sequence(aggregated_list, batch_first=True)

        face_attention_mask = torch.zeros((len(aggregated_list), max_length))
        for data_idx, dur in enumerate(length_list):
            face_attention_mask[data_idx, :dur] = 1

    # subkey = 4 #Face_image
    # aggregated_list = [d[key][subkey] for d in batch if d[key][subkey] is not False]
    # if len(aggregated_list) > 0:
    #     length_list = [len(d) for d in aggregated_list]
    #     max_length = max(length_list)
    #     if max_length>150:
    #         # print("Here length was {}".format(length_list))
    #         max_length = 150
    #         aggregated_list = [i[:max_length] for i in aggregated_list]
    #     aggregated_batch[key][subkey] = torch.nn.utils.rnn.pad_sequence(aggregated_list, batch_first=True)
    #
    #     face_attention_mask = torch.zeros((len(aggregated_list), max_length ))
    #     for data_idx, dur in enumerate(length_list):
    #         face_attention_mask[data_idx, :dur] = 1

    # total_wav = []
    # total_vid = []
    # total_lab = []
    # total_dur = []
    # total_utt = []
    #
    # for cur_batch in batch:
    #     total_wav.append(torch.Tensor(cur_batch[0]))
    #     total_vid.append(torch.Tensor(cur_batch[1]))
    #     total_lab.append(cur_batch[2])
    #     total_dur.append(cur_batch[3])
    #
    #     total_utt.append(cur_batch[4])
    #     # print(total_utt)
    #
    # total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    # total_vid = nn.utils.rnn.pad_sequence(total_vid, batch_first=True)
    #
    # total_lab = torch.Tensor(total_lab)
    # max_dur = np.max(total_dur)
    # attention_mask = torch.zeros(total_wav.shape[0], max_dur)
    # for data_idx, dur in enumerate(total_dur):
    #     attention_mask[data_idx, :dur] = 1
    ## compute mask

    return aggregated_batch


class CramedD_Dataloader:

    def __init__(self, config):
        """
        :param config:
        """
        self.config = config

        dataset_train, dataset_val, dataset_test = self._get_datasets()

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)

        # os.system("taskset -c -p 0-95 %d" % os.getpid())

        num_cores = len(os.sched_getaffinity(0)) - 1
        # num_cores = int(len(os.sched_getaffinity(0))/2)

        num_cores = 0

        print("Available cores {}".format(len(os.sched_getaffinity(0))))
        print("We are changing dataloader workers to num of cores {}".format(num_cores))

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

        # self.total_loader = torch.utils.data.DataLoader(dataset_total,
        #                                                 batch_size=self.config.training_params.test_batch_size,
        #                                                shuffle=False,
        #                                                 num_workers=self.config.training_params.data_loader_workers,
        #                                                pin_memory=self.config.training_params.pin_memory)

    def _get_datasets(self):

        train_dataset = CremadDataset(config=self.config, mode="train")
        valid_dataset = CremadDataset(config=self.config, mode="val")
        test_dataset = CremadDataset(config=self.config, mode="test")
        # total_dataset = CremadDataset(config=self.config, mode="total")

        return train_dataset, valid_dataset, test_dataset


# a = CremadDataset(config = {}, visual_path = "project_default_directory/CremaD/CREMA-D",
#                   audio_path="project_default_directory/CremaD/CREMA-D/AudioWAV", fps=1, mode="train")
# b = a.__getitem__(0)
# import matplotlib.pyplot as plt
# import einops
# plt.imshow(einops.rearrange(b["data"][1][0],"a b c -> b c a").squee


# save_dir = "project_default_directory/data/2023_data/CREMAD_models/unimodal_audio_VAVL_linearcls_fold0.pth.tar"
# checkpoint = torch.load(save_dir, map_location="cpu")
# new_checkpoint = {}
# new_checkpoint["logs"] = checkpoint["logs"]
# new_checkpoint["optimizer_state_dict"] = checkpoint["optimizer_state_dict"]
# new_checkpoint["scheduler_state_dict"] = checkpoint["scheduler_state_dict"]
# new_checkpoint["configs"] = checkpoint["configs"]
#
# new_best_model_state_dict = {".".join(i.split(".")[1:]):checkpoint["model_state_dict"][i]  for i in checkpoint["best_model_state_dict"]}
# new_model_state_dict = {".".join(i.split(".")[1:]):checkpoint["model_state_dict"][i]  for i in checkpoint["model_state_dict"]}
# new_checkpoint["best_model_state_dict"] = new_best_model_state_dict
# new_checkpoint["model_state_dict"] = new_model_state_dict
# new_save_dir = 'project_default_directory/data/2023_data/CREMAD_models/unimodal_audio_VAVL_linearcls_nm_fold0.pth.tar'
# torch.save(new_checkpoint, new_save_dir)

# check if two lists contain any same elements
# a = [1,2,3]
# b = [4,5,6]
# check it
# print(set(a).isdisjoint(b))

# remove duplicated
