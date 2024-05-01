import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomFeatureExtractor2D(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        shape_os_wind = observation_space['wind'].shape
        shape_os_bp = observation_space['pos'].shape
        shape_os_gp = observation_space['obj'].shape

        min_d = min(shape_os_wind[1:])
        cn = []
        chs = 1
        kernel_size=(3,3)

        k=0
        while min_d > 4 and k < 5:
            cn.append(nn.Conv2d(chs, 2*chs, kernel_size, 2, 1))
            cn.append(nn.MaxPool2d(kernel_size, 2, 1))
            cn.append(nn.ReLU())
            min_d = min_d//4
            chs = 2*chs
        
        self.conv_net = nn.ModuleList(cn)

        
        shp = torch.zeros(shape_os_wind)
        for l in self.conv_net:
            shp = l(shp)

        opt_cnn = len(shp.flatten())
        
        n_bp = len(torch.zeros(shape_os_bp).flatten())
        n_op = len(torch.zeros(shape_os_gp).flatten())

        self.fc_net = nn.ModuleList([
            nn.Linear(opt_cnn + n_bp + n_op, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            ])

    def forward(self, x):
        wnds = x["wind"]
        bps = x["pos"]
        op = x["obj"]
        for l in self.conv_net:
            wnds = l(wnds)
        wnds = wnds.view(wnds.size(0), -1)
        bps = bps.view(bps.size(0), -1)
        op = op.view(op.size(0), -1)
        x = torch.cat((wnds, bps, op), dim=1)
        for l in self.fc_net:
            x = l(x)
        return x
    