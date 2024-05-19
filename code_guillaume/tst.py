import torch
import torch.nn as nn
from gym import spaces
import time
import random
from tqdm import tqdm

# print(torch.cuda.is_available())

# print(torch.cuda.device_count(), torch.cuda.get_device_name())

# b = 5000
# n = 1000

# # def timing(device, dsp=True):
# #     t0 = time.time()

# #     t = torch.rand((b,20)).to(device)
# #     l = nn.Sequential(nn.Linear(20, 15).to(device), nn.ReLU().to(device), nn.Linear(15,10).to(device), nn.ReLU().to(device), nn.Linear(10,1).to(device), nn.Softmax(dim=-1).to(device), nn.Linear(1, 10).to(device), nn.ReLU().to(device), nn.Linear(10, 15).to(device), nn.ReLU().to(device), nn.Linear(15,20).to(device), nn.ReLU().to(device))

# #     t1 = time.time()

# #     for i in tqdm(range(n)) if dsp else range(n):
# #         t = l(t)

# #     t2 = time.time()

# #     return [t2-t0, t2-t1, t1-t0]

# # for d in ["cpu", "cuda", "cuda", "cpu", "cpu", "cuda"]:
# #     print(d)
# #     print(timing(d, dsp=False))

# class TestModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv = nn.Sequential(nn.Conv3d(2, 10, 3, 1, 1), nn.MaxPool3d(3, 1, 1), nn.ReLU(),nn.Conv3d(10, 10, 3, 1, 1), nn.MaxPool3d(3, 1, 1), nn.ReLU(),nn.Conv3d(10, 2, 3, 1, 1), nn.MaxPool3d(3, 1, 1), nn.ReLU(), nn.Flatten())
#         self.layers = nn.Sequential(nn.Linear(2*20*20*20, 15), nn.ReLU(), nn.Linear(15,10), nn.ReLU(), nn.Linear(10,1), nn.Softmax(dim=-1))

#     def forward(self, x):
#         return self.layers(self.conv(x))


# m = TestModel().cuda()

# t = torch.rand((1000,2,20,20,20)).to("cuda")

# print(m(t))
# # print(t)

import os 
import numpy
import matplotlib.pyplot as plt

from pathlib import Path

# path = 'C:/Users/guill/Documents/Python Scripts/PSC/Tests_RL/code_guillaume/trash_chckpt/chckpt_input_4'
# sep="-"
# dev = 1

path = "C:/Users/guill/Documents/Python Scripts/PSC/Tests_RL/code_guillaume/ddpg_multiagent_3d_wt_obs/5"
sep="_"
dev = .01

# path = 'C:/Users/guill/Documents/Python Scripts/PSC/Tests_RL/code_guillaume/trash_chckpt/chckpt_misc'
# sep="-"

# path = 'C:/Users/guill/Documents/Python Scripts/PSC/Tests_RL/code_guillaume/chckpt'
# sep="-"

weights = sorted(Path(path).iterdir(), key=os.path.getmtime)
weights = os.listdir(path)
scores = [float(str(wght).split(sep)[-1].split(".")[0] + "." + str(wght).split(sep)[-1].split(".")[1]) for wght in weights]
base = [0.093 for weight in weights]
avg = [numpy.average(scores) for weight in weights]
# print(scores)

scores = numpy.array(scores) + dev*numpy.random.normal(size=(len(scores)))

plt.plot(scores, '+')
plt.plot(avg, 'r')
# plt.plot(base, 'g')
plt.show()