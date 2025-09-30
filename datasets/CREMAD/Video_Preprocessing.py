import pickle

import librosa
import pandas as pd
import cv2
import os
import pdb
import numpy as np
from scipy import signal


class videoReader(object):
    def __init__(self, video_path, frame_interval=1, frame_kept_per_second=1):
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.frame_kept_per_second = frame_kept_per_second

        # pdb.set_trace()
        self.vid = cv2.VideoCapture(self.video_path)
        self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))
        self.video_frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_len = int(self.video_frames / self.fps)

    # def video2frame(self, frame_save_path):
    #     self.frame_save_path = frame_save_path
    #     success, image = self.vid.read()
    #     count = 0
    #     while success:
    #         count += 1
    #         if count % self.frame_interval == 0:
    #             save_name = '{}/frame_{}_{}.jpg'.format(self.frame_save_path, int(count / self.fps),
    #                                                     count)  # filename_second_index
    #             cv2.imencode('.jpg', image)[1].tofile(save_name)
    #         success, image = self.vid.read()

    def video2frame_update(self, frame_save_path, min_save_frame):
        self.frame_save_path = frame_save_path

        count = 0
        save_count = 0
        frame_interval = int(self.fps / self.frame_kept_per_second)
        while count < self.video_frames:
            ret, image = self.vid.read()
            if not ret:
                break
            if count % self.fps == 0:
                frame_id = 0
            if frame_id < frame_interval * self.frame_kept_per_second and frame_id % frame_interval == 0:
                save_name = "{0}/{1:05d}.jpg".format(self.frame_save_path, count)
                cv2.imencode(".jpg", image)[1].tofile(save_name)
                save_count += 1

            frame_id += 1
            count += 1

        if save_count < min_save_frame:
            add_count = min_save_frame - save_count
            count = 0
            if self.video_frames < min_save_frame:
                while count < add_count:
                    frame_id = np.random.randint(0, min_save_frame)
                    save_name = "{0}/{1:05d}.jpg".format(self.frame_save_path, frame_id)
                    if not os.path.exists(save_name):
                        self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, image = self.vid.read()
                        cv2.imencode(".jpg", image)[1].tofile(save_name)
                        count += 1
            else:
                while count < add_count:
                    frame_id = np.random.randint(0, self.video_frames)
                    save_name = "{0}/{1:05d}.jpg".format(self.frame_save_path, frame_id)
                    if not os.path.exists(save_name):
                        self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                        ret, image = self.vid.read()
                        cv2.imencode(".jpg", image)[1].tofile(save_name)
                        count += 1


class CRAMED_dataset(object):
    def __init__(
        self,
        path_to_dataset="project_default_directory/CremaD/CREMA-D",
        frame_interval=1,
        frame_kept_per_second=1,
    ):
        self.path_to_video = os.path.join(path_to_dataset, "VideoFlash")
        self.path_to_audio = os.path.join(path_to_dataset, "AudioWAV")
        self.frame_kept_per_second = frame_kept_per_second
        self.sr = 16000

        self.path_to_save = os.path.join(path_to_dataset, "Image-{:02d}-FPS".format(self.frame_kept_per_second))
        if not os.path.exists(self.path_to_save):
            os.mkdir(self.path_to_save)

        self.path_to_save_audio = os.path.join(path_to_dataset, "Audio-{:d}".format(1004))
        if not os.path.exists(self.path_to_save_audio):
            os.mkdir(self.path_to_save_audio)

        csv_file = pd.read_csv(os.path.join(path_to_dataset, "processedResults/summaryTable.csv"))
        self.file_list = list(csv_file["FileName"])

    def extractImage(self):

        for each_video in self.file_list:
            print("Precessing {} ...".format(each_video))
            video_dir = os.path.join(self.path_to_video, each_video + ".flv")
            self.videoReader = videoReader(video_path=video_dir, frame_kept_per_second=self.frame_kept_per_second)

            save_dir = os.path.join(self.path_to_save, each_video)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            self.videoReader.video2frame_update(frame_save_path=save_dir, min_save_frame=3)  # 每个视频最少取三张图片

    def extractWav(self):
        for each_audio in self.file_list:
            print("Precessing {} ...".format(each_audio))
            audio_dir = os.path.join(self.path_to_audio, each_audio + ".wav")

            samples, rate = librosa.load(audio_dir, sr=self.sr)
            resamples = np.tile(samples, 10)[: self.sr * 10]
            resamples[resamples > 1.0] = 1.0
            resamples[resamples < -1.0] = -1.0

            # spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
            frequencies, times, spectrogram = signal.spectrogram(resamples, rate, nperseg=512, noverlap=353)
            spectrogram = np.log(np.abs(spectrogram) + 1e-7)
            # mean = np.mean(spectrogram)
            # std = np.std(spectrogram)
            # spectrogram = np.divide(spectrogram - mean, std + 1e-9)
            # print(spectrogram.shape)
            save_name = os.path.join(self.path_to_save_audio, each_audio + ".pkl")
            with open(save_name, "wb") as fid:
                pickle.dump(spectrogram, fid)


# cramed = CRAMED_dataset()
# cramed.extractImage()
# cramed.extractWav()
#
#
# from sklearn.model_selection import KFold
# import numpy as np
# def random_split_list(input_list, num_splits):
#     kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)
#     kf_val = KFold(n_splits=num_splits, shuffle=True, random_state=42)
#     folds = {}
#     for i, (train_index, test_index) in enumerate(kf.split(input_list)):
#         print(i)
#         split_trainval = np.take(input_list, train_index)
#         for j, (train_index, val_index) in enumerate(kf_val.split(split_trainval.tolist())):
#             split_train = np.take(split_trainval.tolist(), train_index)
#             split_val = np.take(split_trainval.tolist(), val_index)
#             break
#         split_test = np.take(input_list, test_index)
#         folds[i] = {"train": split_train.tolist(), "val": split_val.tolist(), "test":split_test.tolist()}
#     return folds
#
# # Example usage:
# num_splits = 10
# all_names = np.unique(np.array([ i.split("_")[0] for i in os.listdir("project_default_directory/CremaD/CREMA-D/AudioWAV") ]))
# folds = random_split_list(all_names, num_splits)
# print(folds.keys())
# for f in folds:
#     for set in folds[f]:
#         this_set_list = []
#         for name in folds[f][set]:
#             for i in os.listdir("project_default_directory/CremaD/CREMA-D/VideoFlash"):
#                 if name in i: this_set_list.append(i)
#         folds[f][set] = this_set_list
#
# message = "Total Folds \n"
# for f in folds:
#     message += "fold {} ".format(f)
#     message += "train {}/ val {}/ test {} with total {}\n".format(len(folds[f]["train"]),len(folds[f]["val"]),len(folds[f]["test"]), len(folds[f]["train"])+len(folds[f]["val"])+len(folds[f]["test"]))
# print(message)
#
# with open('./datasets/CREMAD/data_splits_persubj.pkl', "w") as json_file:
#     json.dump(folds, json_file)
