import os
import pickle

import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from tqdm import tqdm


def extract_frames_and_audio(video_paths, labels, num):
    video_path = video_paths[num]
    label = labels[num]
    audio_fps = 16000
    # cap = AVIReader(video_path)
    # total_frames = cap.num_frames
    # frame_fps = cap.fps

    # if not cap.isOpened():
    #     raise Exception("Error: Could not open the video file.")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_fps = int(cap.get(cv2.CAP_PROP_FPS))

    video_clip = VideoFileClip(video_path)
    if video_clip.audio is None:
        audio_clip = np.empty(shape=1)
    else:
        audio_clip = video_clip.audio.to_soundarray(fps=audio_fps)[:, 0]

    frames, frame_number = [], 0
    while frame_number < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_number % frame_fps == 0:
            frames.append(frame)
        frame_number += 1
    cap.release()
    frames = np.array(frames)
    return {
        video_path.split("/")[-1]: {
            "data": {"vision": frames, "audio": audio_clip},
            "label": label,
        }
    }


data_roots = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/UCF"

annotation_folder = os.path.join(data_roots, "Split_kkontras")
annotation_train = os.path.join(annotation_folder, "trainlist01.txt")
annotation_val = os.path.join(annotation_folder, "testlist01.txt")
annotation_test = os.path.join(os.path.join(data_roots, "ucfTrainTestlist"), "testlist01.txt")

class_to_int = {}
with open(
    ".datasets/UCF/ucfTrainTestlist/classInd.txt",
    "r",
) as file:
    for line in file:
        values = line.replace("\n", "").strip().split(" ")
        class_to_int[values[1]] = values[0]


train_data, train_labels = [], []

with open(annotation_train, "r") as file:
    for line in file:
        values = line.replace("\n", "").strip().split(" ")
        train_data.append(os.path.join(os.path.join(data_roots, "UCF-101"), values[0]))
        train_labels.append(values[1])
train_labels = np.array(train_labels).astype(int)

val_data, val_labels = [], []
with open(annotation_val, "r") as file:
    for line in file:
        values = line.replace("\n", "").strip().split(" ")
        val_data.append(os.path.join(os.path.join(data_roots, "UCF-101"), values[0]))
        val_labels.append(values[1])
val_labels = np.array(val_labels).astype(int)

test_data, test_labels = [], []
with open(annotation_test, "r") as file:
    for line in file:
        values = line.replace("\n", "").strip().split(" ")
        test_data.append(os.path.join(os.path.join(data_roots, "UCF-101"), values[0]))
        test_labels.append(class_to_int[values[0].split("/")[0]])
test_labels = np.array(test_labels).astype(int)

metrics_scrumble = [extract_frames_and_audio(train_data, train_labels, num) for num in tqdm(range(len(train_labels)), "Mean-STD Calculating ")]
total_train_data = {i: res[i] for res in metrics_scrumble for i in res}

metrics_scrumble = [extract_frames_and_audio(val_data, val_labels, num) for num in tqdm(range(len(val_labels)), "Mean-STD Calculating ")]
total_val_data = {i: res[i] for res in metrics_scrumble for i in res}

metrics_scrumble = [extract_frames_and_audio(test_data, test_labels, num) for num in tqdm(range(len(test_labels)), "Mean-STD Calculating ")]
total_test_data = {i: res[i] for res in metrics_scrumble for i in res}


total_data = {}
total_data.update(total_train_data)
total_data.update(total_val_data)
total_data.update(total_test_data)


with open(os.path.join(data_roots, "UCF_fps1_audio16000.pkl"), "wb") as handle:
    pickle.dump(total_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
