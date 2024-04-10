import csv
import cv2
import threading
import numpy as np

import liblmp

layout: str = "../layout/Unicorn.lmpl6r"
empty_pattern: str = "../layout/Unicorn.lmp6r"

layout_file = open(layout, "r", encoding="shift-jis")
empty_pattern_file = open(empty_pattern, "r", encoding="shift-jis")
layout_reader = csv.reader(layout_file, delimiter="\t")
empty_pattern_reader = csv.reader(empty_pattern_file, delimiter="\t")
layout_list = [row for row in layout_reader]
empty_pattern_list = [row for row in empty_pattern_reader]
layout_file.close()
empty_pattern_file.close()

layout_data = liblmp.parse_lmp(layout_list)
pattern_data = liblmp.parse_lmp(empty_pattern_list)

led_df = layout_data["LAMP"].merge(layout_data["KINDPOSTFIX"], on="Kind")

pattern_data["PATTERN"].loc[0] = [0, "BASE", 0, 1000, 0, "", 60, -1, 500, "TRUE", "test pattern", "FALSE", 0]
video = cv2.VideoCapture("../data/dataset.mp4")
start_time: int = 75 * 1000
end_time: int = 75 * 1000 + 1000
duration = end_time - start_time
fps = 30
for i, row in led_df.iterrows():
    led_name = row["LabelBase"] + row["Postfix"]
    pattern_data["LAYER"].loc[i] = [i, 0, led_name, "", "TRUE", "FALSE"]
    pattern_data["CLIP"].loc[i] = [i * 2, i, 0, duration, duration, 2, ""]

def sample_led(pixel: any, postfix: str) -> int:
    value = 0
    if postfix == "_B":
        value = pixel[0]
    if postfix == "_G":
        value = pixel[1]
    if postfix == "_R":
        value = pixel[2]
    if postfix == "_A":
        value = 255
    return int(value)

def mean_sample_led(frame: cv2.typing.MatLike, x: int, y: int, postfix: str) -> int:
    pixel_delta = [(0, 0), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    pixel = np.zeros((3))
    for d in pixel_delta:
        pixel += frame[y + d[0], x + d[1]]
    pixel /= 9
    return sample_led(pixel, postfix)

frame_idx = 0
frame_len = int(duration * (fps / 1000) + 1)
frame_duration = int(1000 / fps)
clip_data = np.zeros((len(led_df), frame_len))

pixel_max = np.zeros(4)
pixel_min = np.ones(4) * 255

led_kinds = ["_B", "_G", "_R", "_A"]
for frame_idx in range(0, frame_len):
    frame_time = frame_idx * frame_duration
    video.set(cv2.CAP_PROP_POS_MSEC, start_time + frame_time)
    _, frame = video.read()
    for i in range(3):
        for j, row in led_df[led_df["Postfix"] == led_kinds[i]].iterrows():
            value = mean_sample_led(frame, int(row["X"]), int(row["Y"]), row["Postfix"])
            pixel_max[i] = max(pixel_max[i], value)
            pixel_min[i] = min(pixel_min[i], value)

pixel_coef = 1.0 / (pixel_max - pixel_min)
pixel_coef[3] = 1.0 / 255
pixel_min[3] = 0

print("Normalized.")
print("max: ", pixel_max)
print("min: ", pixel_min)
print("range: ", pixel_coef)

for frame_idx in range(0, frame_len):
    frame_time = frame_idx * frame_duration
    video.set(cv2.CAP_PROP_POS_MSEC, start_time + frame_time)
    _, frame = video.read()
    
    def add_keyframe(i, row):
        value = mean_sample_led(frame, int(row["X"]), int(row["Y"]), row["Postfix"])
        led_idx = led_kinds.index(row["Postfix"])
        value = int((value - pixel_min[led_idx]) * pixel_coef[led_idx] * 255)
        pattern_data["CLIP"].iat[i, 6] += "{},{};".format(frame_time, value)
        clip_data[i][frame_idx] = value

    # シングルスレッド
    # for i, row in led_df.iterrows():
    #     add_keyframe(i, row)

    threads = []
    for i, row in led_df.iterrows():
        t = threading.Thread(target=add_keyframe, args=(i, row))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


output_file = open("./output.lmp6r", "w", encoding="shift-jis", newline="\n")
output_data = liblmp.export_lmp6r(pattern_data)
output_file.write(output_data)
output_file.close()

print(clip_data)
