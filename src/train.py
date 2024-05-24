import argparse
import numpy as np
import cv2
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch import optim

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
parser.add_argument("output_path")
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


def make_input(led: torch.Tensor, spec: torch.Tensor, lcd :torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


led_train, spec_train, lcd_train, led_t_train = led_train.cuda(), spec_train.cuda(), lcd_train.cuda(), led_t_train.cuda()

for i in range(1, train_data_num + 1):
    print("load dataset {}".format(i))
    led, spec, lcd = load_dataset(args.dataset_path, i)
    frame_len = led.shape[0]
    sample_len = frame_len - n_frames

    led_t, spec_t, lcd_t = make_input(led, spec, lcd)
    led_t_t = led[n_frames:n_frames + sample_len].cuda()

    led_train = torch.cat([led_train, led_t], 0)
    spec_train = torch.cat([spec_train, spec_t], 0)
    lcd_train = torch.cat([lcd_train, lcd_t], 0)
    led_t_train = torch.cat([led_t_train, led_t_t], 0)

print("led:\t", led_train.shape)
print("spec:\t", spec_train.shape)
print("lcd:\t", lcd_train.shape)
print("led_t:\t", led_t_train.shape)
print("VRAM Use:\t{} MB".format(torch.cuda.memory_allocated(0) / 1000 / 1000))

dataset = TensorDataset(led_train, spec_train, lcd_train, led_t_train)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.spec_conv = nn.Conv1d(
            n_frames, n_frames, 5
        )
        self.c1_1 = nn.Conv1d(n_frames - 1, 1, 3, 2)
        self.c1_2 = nn.Conv1d(n_frames, 1, 3, 1)
        self.c2 = nn.Conv3d(n_frames, 4, (3, 3, 1), (2, 2, 1)) 
        self.c3 = nn.Conv3d(4, 1, (3, 3, 1), (2, 2, 1))
        self.c4 = nn.Conv3d(1, 1, (3, 3, 1), (2, 2, 1))
        #self.lstm = nn.LSTM(input_size=583, hidden_size=64, batch_first=True)
        self.lstm = nn.LSTM(input_size=267, hidden_size=267, batch_first=True)
        self.lin = nn.Linear(267, 267)
        self.lin0 = nn.Linear(450, 267)
        self.relu = nn.ReLU()
    
    def forward(self, led, spec, lcd):
        cled = self.c1_1(led)
        cspec = self.c1_2(spec)
        clcd1 = self.c2(lcd)
        clcd2 = self.c3(clcd1)
        ylcd = torch.reshape(clcd2, clcd2.shape[:1] + (-1,))
        yspec = torch.reshape(cspec, cspec.shape[:1] + (-1,))
        yled = torch.reshape(cled, cled.shape[:1] + (-1,))
        y = torch.cat([ylcd, yspec], 1)
        y = self.lin0(y)
        y, h = self.lstm(y, None)
        y = self.relu(y)
        y = self.lin(y)
        return y

net = Net()
net.cuda()
print(net)

loss_fnc = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)
record_loss_train = []
epochs = 200
for i in range(epochs):
    net.train()
    loss_train = 0
    for j, (led, spec, lcd, t) in enumerate(loader):
        led, spec, lcd, t = led.cuda(), spec.cuda(), lcd.cuda(), t.cuda()
        y = net(led, spec, lcd)
        loss = loss_fnc(y, t)
        loss_train += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_train /= j + 1
    record_loss_train.append(loss_train)
    print("epoch:\t{}\tloss_train={}".format(i, loss))

test_id = 17
test_led, test_spec, test_lcd = make_input(*load_dataset(args.dataset_path, test_id))
net.eval()
output: torch.Tensor = net(test_led, test_spec, test_lcd)
np.savetxt("output.csv", output.cpu().detach().numpy(), fmt="%d")

# Export the model   
torch.onnx.export(net,         # model being run 
        (torch.zeros(1, n_frames - 1, n_leds).cuda(), torch.zeros(1, n_frames, n_spec).cuda(), torch.zeros(1, n_frames, lcd_fixed_shape[0], lcd_fixed_shape[1], 3).cuda()),       # model input (or a tuple for multiple inputs) 
        "Network.onnx",       # where to save the model  
        export_params=True,  # store the trained parameter weights inside the model file 
        opset_version=11,    # the ONNX version to export the model to 
        do_constant_folding=True,  # whether to execute constant folding for optimization 
        input_names = ['input'],   # the model's input names 
        output_names = ['output']) # the model's output names 