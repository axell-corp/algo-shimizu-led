import csv
import cv2
import threading
import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
import pandas as pd
import warnings

import liblmp

dataset_path = "../data/dataset_test"
led_raw_filename = dataset_path + "/led/led_raw_{}.csv"
led_filename = dataset_path + "/led/led_{}.csv"
spec_filename = dataset_path + "/spc/spec_{}.png"
spec_png_filename = dataset_path + "/spc/spec_{}.csv"
lcd_filename = dataset_path + "/lcd/lcd_{}.mp4"
os.makedirs(dataset_path + "/lcd", exist_ok=True)
os.makedirs(dataset_path + "/led", exist_ok=True)
os.makedirs(dataset_path + "/spc", exist_ok=True)

def make_spect(audiofile_name: str, data_id: int):
    audio_y, sr = librosa.load(audiofile_name)
    melspect = librosa.feature.melspectrogram(y=audio_y, n_fft=2048, hop_length=int(sr/30))
    melspect = librosa.power_to_db(melspect, ref=np.max)
    librosa.display.specshow(melspect, x_axis="time", y_axis="mel", sr=sr)
    plt.savefig(spec_filename.format(data_id))
    np.savetxt(spec_png_filename.format(data_id), melspect)


def make_led_lcd(videofile_name: str, layout_data: dict[str, pd.DataFrame], data_id: int):
    video = cv2.VideoCapture(videofile_name)
    video.set(cv2.CAP_PROP_FPS, 30)
    fps = 30

    lcd_tl = np.array((108, 319))
    lcd_br = np.array((544, 639))
    lcd_shape = lcd_br - lcd_tl
    lcd_video_writer = cv2.VideoWriter(lcd_filename.format(data_id), cv2.VideoWriter_fourcc(*"mp4v"), fps, lcd_shape)

    pixel_max = np.ones(4) * 255
    pixel_min = np.zeros(4)
    pixel_coef = 1.0 / (pixel_max - pixel_min)
    pixel_coef[3] = 1.0 / 255
    pixel_min[3] = 0
    # 60fps->30fps
    frame_len = int((video.get(cv2.CAP_PROP_FRAME_COUNT) + 1) / 2)
    led_df = layout_data["LAMP"].merge(layout_data["KINDPOSTFIX"], on="Kind").query("Postfix != '_A'")
    clip_data = np.zeros((len(led_df), frame_len), dtype=np.int32)
    for frame_idx in range(0, frame_len):
        # 60fps->30fps
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx * 2)
        _, frame = video.read()
        lcd_frame = frame.copy()[lcd_tl[1]:lcd_br[1], lcd_tl[0]:lcd_br[0], :]
        lcd_video_writer.write(lcd_frame)
        
        def add_keyframe(i, row):
            value = liblmp.mean_sample_led(frame, int(row["X"]), int(row["Y"]), row["Postfix"])
            led_idx = liblmp.led_kinds.index(row["Postfix"])
            value = int((value - pixel_min[led_idx]) * pixel_coef[led_idx] * 255)
            value = max(min(value, 255), 0)
            clip_data[i][frame_idx] = value

        # シングルスレッド
        # for i, row in led_df.iterrows():
        #     add_keyframe(i, row)

        threads = []
        i = 0
        for _, row in led_df.iterrows():
            t = threading.Thread(target=add_keyframe, args=(i, row))
            t.start()
            threads.append(t)
            i += 1
        for t in threads:
            t.join()

    lcd_video_writer.release()
    np.savetxt(led_raw_filename.format(data_id), clip_data, fmt="%d")

layout: str = "../layout/Unicorn2.lmpl6r"
layout_file = open(layout, "r", encoding="shift-jis")
layout_reader = csv.reader(layout_file, delimiter="\t")
layout_list = [row for row in layout_reader]
layout_file.close()
layout_data = liblmp.parse_lmp(layout_list)

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

dataset_num = 3

for video_id in range(1, dataset_num + 1):
    video_file = "../data/videos/{}.mp4".format(video_id)
    print(video_file)
    make_spect(video_file, video_id)
    make_led_lcd(video_file, layout_data, video_id)

led_array = np.loadtxt(led_raw_filename.format(1), np.int32)
print(led_array.shape)
led_max = np.ones((led_array.shape[0])) * np.inf * -1
led_min = np.ones((led_array.shape[0])) * np.inf
led_max = np.maximum(led_max, np.max(led_array, axis = 1))
led_min = np.minimum(led_min, np.min(led_array, axis = 1))
for data_id in range(2, dataset_num + 1):
    led_array = np.loadtxt(led_raw_filename.format(data_id), np.int32)
    led_max = np.maximum(led_max, np.max(led_array, axis = 1))
    led_min = np.minimum(led_min, np.min(led_array, axis = 1))
print(led_max)
print(led_min)

led_range = led_max - led_min
led_range = np.maximum(led_range, np.ones(led_range.shape))

for data_id in range(1, dataset_num + 1):
    led_array = np.loadtxt(led_raw_filename.format(data_id), np.int32)
    for row in range(led_array.shape[1]):
        led_array[:, row] = (led_array[:, row] - led_min) / led_range * 255
    np.savetxt(led_filename.format(data_id), led_array, fmt="%d")
