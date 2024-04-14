import torch
from gym import spaces

t = torch.rand((10,5)).to("cpu").view((5,10))
print(t)