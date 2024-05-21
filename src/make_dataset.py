import numpy as np
import cv2

def make_dataset(data_id: int, lmp_fmt: str, spec_fmt: str, lcd_fmt: str):
    led = np.loadtxt(lmp_fmt.format(data_id), dtype=np.int32)
    spec = np.loadtxt(spec_fmt.format(data_id), dtype=np.float32)
    lcd = cv2.VideoCapture(lcd_fmt.format(data_id))
    lcd_shape = (int(lcd.get(cv2.CAP_PROP_FRAME_WIDTH)), int(lcd.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(lcd.get(cv2.CAP_PROP_FRAME_COUNT))
    print(led.shape)
    print(spec.shape)

make_dataset(1, "../data/dataset/led/led_{}.csv", "../data/dataset/spc/spec_{}.csv", "../data/dataset/lcd/lcd_{}.mp4")
