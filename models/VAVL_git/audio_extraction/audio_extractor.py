"""
Author: Lucas Goncalves
2023

"""

import os

from moviepy.editor import VideoFileClip
from pydub import AudioSegment

root = "/path_to_data_dir/data/Dataset_Name/"

input_directory = root
output_directory = root + "Audios"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for filename in os.listdir(input_directory):
    if filename.endswith(".mp4") or filename.endswith(".flv"):
        video = VideoFileClip(os.path.join(input_directory, filename))
        video.audio.write_audiofile(os.path.join(output_directory, filename.rsplit(".", 1)[0] + ".wav"))

for filename in os.listdir(output_directory):
    if filename.endswith(".wav"):
        audio = AudioSegment.from_wav(os.path.join(output_directory, filename))
        audio = audio.set_frame_rate(16000)  # Set frame rate to 16kHz
        audio = audio.set_channels(1)  # Set to mono channel
        audio.export(os.path.join(output_directory, filename), format="wav")  # Export audio
