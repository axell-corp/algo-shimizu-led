import argparse
import numpy as np
import cv2
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch import optim
import onnx
import onnxruntime

# returns (led, spectrogram, lcd)
def load_dataset(dataset_path: str, data_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    led_path = "{}/led/led_{}.csv".format(dataset_path, data_id)
    spec_path = "{}/spc/spec_{}.csv".format(dataset_path, data_id)
    lcd_path = "{}/lcd/lcd_{}.mp4".format(dataset_path, data_id)
    led = np.loadtxt(led_path, dtype=np.int32)
    spec = np.loadtxt(spec_path, dtype=np.float32)
    lcd_video = cv2.VideoCapture(lcd_path)
    lcd_width = int(lcd_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    lcd_height = int(lcd_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    lcd_frames = int(lcd_video.get(cv2.CAP_PROP_FRAME_COUNT))
    lcd = np.zeros((lcd_frames, lcd_height, lcd_width, 3))
    for i in range(lcd_frames):
        lcd_video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, lcd_f = lcd_video.read()
        if not ok:
            print("video read error occured. file={}, frame={}".format(lcd_path, i))
        lcd[i, :, :, :] = lcd_f
    return torch.tensor(led.T), torch.tensor(spec.T), torch.tensor(lcd)

    

parser = argparse.ArgumentParser()
parser.add_argument("dataset_path")
args = parser.parse_args()
n_frames = 10
train_data_num = 10

first_led, first_spec, first_lcd = load_dataset(args.dataset_path, 1)
_, lcd_height, lcd_width, _ = first_lcd.shape
lcd_fixed_shape = (int(lcd_height / 8), int(lcd_width / 8))
n_leds = first_led.shape[1]
n_spec = first_spec.shape[1]
led_train = torch.zeros((0, n_frames - 1, first_led.shape[1]))
spec_train = torch.zeros((0, n_frames, first_spec.shape[1]))
lcd_train = torch.zeros((0, n_frames, lcd_fixed_shape[0], lcd_fixed_shape[1], 3))
led_t_train = torch.zeros((0, n_leds))
first_led, first_spec, first_lcd = None, None, None


def make_input(led: torch.Tensor, spec: torch.Tensor, lcd :torch.Tensor, frame_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sample_len = frame_len - n_frames
    led_t = torch.zeros((sample_len, n_frames - 1, led.shape[1]))
    spec_t = torch.zeros((sample_len, n_frames, spec.shape[1]))
    lcd_t = torch.zeros((sample_len, n_frames, lcd_height, lcd_width, 3))
    for j in range(sample_len):
        led_t[j] = led[j:j+n_frames-1, :]
        spec_t[j] = spec[j:j+n_frames, :]
        lcd_t[j] = lcd[j:j+n_frames, :, :, :]
    fixed_lcd_t = nn.functional.interpolate(lcd_t.cuda(), (lcd_fixed_shape[0], lcd_fixed_shape[1], 3))
    return led_t.cuda(), spec_t.cuda(), fixed_lcd_t


loss_fnc = nn.MSELoss()
record_loss_train = []

test_id = 18
test_led, test_spec, test_lcd = load_dataset(args.dataset_path, test_id)
test_led, test_spec, test_lcd = make_input(test_led, test_spec, test_lcd, test_led.shape[0])

session = onnxruntime.InferenceSession("./output.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
input0_name = session.get_inputs()[0].name
input1_name = session.get_inputs()[1].name
output_name = session.get_outputs()[0].name
print(input0_name)
print(input1_name)
print(output_name)
y = session.run([output_name], {input0_name: test_spec.cpu().detach().numpy(), input1_name: test_lcd.cpu().detach().numpy()})
y = np.array(y, dtype=np.int32)
y = y.reshape(y.shape[1:])
np.savetxt("output.csv", y.T, fmt="%d")

# output: torch.Tensor = net(test_led, test_spec, test_lcd)
# np.savetxt("output.csv", output.cpu().detach().numpy(), fmt="%d")