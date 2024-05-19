import torch
import numpy as np
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomActorNetwork(nn.Module):
    def __init__(self, observation_space, action_space, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        shape_os_wind = observation_space[0].shape
        shape_os_bp = observation_space[1].shape
        shape_as = action_space.shape

        min_d = min(shape_os_wind[1:])
        cn = []
        chs = 2
        kernel_size=(3,3,3)

        while min_d > 8:
            cn.append(nn.Conv3d(chs, 2*chs, kernel_size, 2, 1))
            cn.append(nn.MaxPool3d(kernel_size, 2, 1))
            min_d = min_d//4
            chs = 2*chs
        
        self.conv_net = nn.ModuleList(cn)

        shp = torch.zeros(shape_os_wind)
        for l in self.conv_net:
            shp = l(shp)
        shp = shp.shape

        opt_cnn = 1
        for q in shp:
            opt_cnn *= q
        
        n_bp = 1
        for q in shape_os_bp:
            n_bp *= q
        
        n_as = 1
        for q in shape_as:
            n_as *= q

        self.fc_net = nn.ModuleList([
            nn.Linear(opt_cnn + n_bp, 128), 
            nn.ReLU(), 
            nn.Linear(128, 128), 
            nn.ReLU(), 
            nn.Linear(128, 64), 
            nn.ReLU(), 
            nn.Linear(64, 32), 
            nn.ReLU(), 
            nn.Linear(32, n_as),
            nn.Tanh()
            ])

    def forward(self, x):
        wnds = torch.stack([s[0] for s in x], 0)
        bps = torch.stack([s[1] for s in x], 0)
        for l in self.conv_net:
            wnds = l(wnds)
        wnds = wnds.view(wnds.size(0), -1)
        bps = bps.view(bps.size(0), -1)
        x = torch.cat((wnds, bps), dim=1)
        for l in self.fc_net:
            x = l(x)
        return x

class CustomCriticNetwork(nn.Module):
    def __init__(self, observation_space, action_space, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        shape_os_wind = observation_space[0].shape
        shape_os_bp = observation_space[1].shape
        shape_as = action_space.shape

        min_d = min(shape_os_wind[1:])
        cn = []
        chs = 2
        kernel_size=(3,3,3)

        while min_d > 8:
            cn.append(nn.Conv3d(chs, 2*chs, kernel_size, 2, 1))
            cn.append(nn.MaxPool3d(kernel_size, 2, 1))
            min_d = min_d//4
            chs = 2*chs
        
        self.conv_net = nn.ModuleList(cn)

        shp = torch.zeros(shape_os_wind)
        for l in self.conv_net:
            shp = l(shp)
        shp = shp.shape

        opt_cnn = 1
        for q in shp:
            opt_cnn *= q
        
        n_bp = 1
        for q in shape_os_bp:
            n_bp *= q
        
        n_as = 1
        for q in shape_as:
            n_as *= q

        self.fc_net = nn.ModuleList([
            nn.Linear(opt_cnn + n_bp + n_as, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            ])

    def forward(self, state, action):
        wnds = torch.stack([s[0] for s in state], 0)
        bps = torch.stack([s[1] for s in state], 0)
        for l in self.conv_net:
            wnds = l(wnds)
        wnds = wnds.view(wnds.size(0), -1)
        bps = bps.view(bps.size(0), -1)
        action = action.view(action.size(0), -1)
        state = torch.cat((wnds, bps, action), dim=1)
        for l in self.fc_net:
            state = l(state)
        return state

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128, device="cpu"):
        super().__init__(observation_space, features_dim)
        shape_os_wind = observation_space["wind"].shape
        shape_os_bp = observation_space["pos"].shape

        min_d = min(shape_os_wind[1:])
        cn = []
        chs = 2
        kernel_size=(3,3,3)

        while min_d > 8:
            cn.append(nn.Conv3d(chs, 2*chs, kernel_size, 2, 1))
            cn.append(nn.ReLU())
            cn.append(nn.MaxPool3d(kernel_size, 2, 1))
            min_d = min_d//4
            chs = 2*chs
        
        self.conv_net = nn.ModuleList(cn).to(device)

        shp = torch.zeros(shape_os_wind)
        for l in self.conv_net:
            shp = l(shp)
        shp = shp.shape

        opt_cnn = 1
        for q in shp:
            opt_cnn *= q
        
        n_bp = 1
        for q in shape_os_bp:
            n_bp *= q
        
        self.fc_net = nn.ModuleList([
            nn.Linear(opt_cnn + n_bp, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, features_dim),
            ]).to(device)

    def forward(self, x):
        wnds = x["wind"]
        bps = x["pos"]
        for l in self.conv_net:
            wnds = l(wnds)
        wnds = wnds.view(wnds.size(0), -1)
        bps = bps.view(bps.size(0), -1)
        x = torch.cat((wnds, bps), dim=1)
        for l in self.fc_net:
            x = l(x)
        return x

# ----------------------------------------------------------------------------------

class CustomActorNetwork2D(nn.Module):
    def __init__(self, observation_space, action_space, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        shape_os_wind = observation_space[0].shape
        shape_os_bp = observation_space[1].shape
        shape_as = action_space.shape

        min_d = min(shape_os_wind[1:])
        cn = []
        chs = shape_os_wind[0]
        kernel_size=(3,3)

        while min_d > 8:
            cn.append(nn.Conv2d(chs, 2*chs, kernel_size, 2, 1))
            cn.append(nn.MaxPool2d(kernel_size, 2, 1))
            min_d = min_d//4
            chs = 2*chs
        
        self.conv_net = nn.ModuleList(cn)

        shp = torch.zeros(shape_os_wind)
        for l in self.conv_net:
            shp = l(shp)
        shp = shp.shape

        opt_cnn = 1
        for q in shp:
            opt_cnn *= q
        
        n_bp = 1
        for q in shape_os_bp:
            n_bp *= q
        
        n_as = 1
        for q in shape_as:
            n_as *= q

        self.fc_net = nn.ModuleList([
            nn.Linear(opt_cnn + n_bp, 128), 
            nn.ReLU(), 
            nn.Linear(128, 128), 
            nn.ReLU(), 
            nn.Linear(128, 64), 
            nn.ReLU(), 
            nn.Linear(64, 32), 
            nn.ReLU(), 
            nn.Linear(32, n_as),
            nn.Tanh()
            ])

    def forward(self, x):
        wnds = torch.stack([s[0] for s in x], 0)
        bps = torch.stack([s[1] for s in x], 0)
        for l in self.conv_net:
            wnds = l(wnds)
        wnds = wnds.view(wnds.size(0), -1)
        bps = bps.view(bps.size(0), -1)
        x = torch.cat((wnds, bps), dim=1)
        for l in self.fc_net:
            x = l(x)
        return x

class CustomCriticNetwork2D(nn.Module):
    def __init__(self, observation_space, action_space, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        shape_os_wind = observation_space[0].shape
        shape_os_bp = observation_space[1].shape
        shape_as = action_space.shape

        min_d = min(shape_os_wind[1:])
        cn = []
        chs = shape_os_wind[0]
        kernel_size=(3,3)

        while min_d > 8:
            cn.append(nn.Conv2d(chs, 2*chs, kernel_size, 2, 1))
            cn.append(nn.MaxPool2d(kernel_size, 2, 1))
            min_d = min_d//4
            chs = 2*chs
        
        self.conv_net = nn.ModuleList(cn)

        shp = torch.zeros(shape_os_wind)
        for l in self.conv_net:
            shp = l(shp)
        shp = shp.shape

        opt_cnn = 1
        for q in shp:
            opt_cnn *= q
        
        n_bp = 1
        for q in shape_os_bp:
            n_bp *= q
        
        n_as = 1
        for q in shape_as:
            n_as *= q

        self.fc_net = nn.ModuleList([
            nn.Linear(opt_cnn + n_bp + n_as, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            ])

    def forward(self, state, action):
        wnds = torch.stack([s[0] for s in state], 0)
        bps = torch.stack([s[1] for s in state], 0)
        action = torch.tensor(action)
        for l in self.conv_net:
            wnds = l(wnds)
        wnds = wnds.view(wnds.size(0), -1)
        bps = bps.view(bps.size(0), -1)
        action = action.view(action.size(0), -1)
        state = torch.cat((wnds, bps, action), dim=1)
        for l in self.fc_net:
            state = l(state)
        return state
    
class CustomFeatureExtractor2D(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        shape_os_wind = observation_space['wind'].shape
        shape_os_bp = observation_space['pos'].shape

        min_d = min(shape_os_wind[1:])
        cn = []
        chs = shape_os_wind[0]
        kernel_size=(3,3)

        k=0
        while min_d > 8 and k < 5:
            cn.append(nn.Conv2d(chs, 2*chs, kernel_size, 2, 1))
            cn.append(nn.MaxPool2d(kernel_size, 2, 1))
            min_d = min_d//4
            chs = 2*chs
        
        self.conv_net = nn.ModuleList(cn)

        shp = torch.zeros(shape_os_wind)
        for l in self.conv_net:
            shp = l(shp)
        shp = shp.shape

        opt_cnn = 1
        for q in shp:
            opt_cnn *= q
        
        n_bp = 1
        for q in shape_os_bp:
            n_bp *= q

        self.fc_net = nn.ModuleList([
            nn.Linear(opt_cnn + n_bp, 128),
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
        for l in self.conv_net:
            wnds = l(wnds)
        wnds = wnds.view(wnds.size(0), -1)
        bps = bps.view(bps.size(0), -1)
        x = torch.cat((wnds, bps), dim=1)
        for l in self.fc_net:
            x = l(x)
        return x
    
class CustomFeatureExtractorWind2D(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        shape_os_wind = observation_space['wind'].shape

        min_d = min(shape_os_wind[1:])
        cn = []
        chs = shape_os_wind[0]
        kernel_size=(3,3)

        while min_d > 8:
            cn.append(nn.Conv2d(chs, 2*chs, kernel_size, 2, 1))
            cn.append(nn.MaxPool2d(kernel_size, 2, 1))
            min_d = min_d//4
            chs = 2*chs
        
        self.conv_net = nn.ModuleList(cn)

        shp = torch.zeros(shape_os_wind)
        for l in self.conv_net:
            shp = l(shp)
        shp = shp.shape

        opt_cnn = 1
        for q in shp:
            opt_cnn *= q
        
        self.fc_net = nn.ModuleList([
            nn.Linear(opt_cnn, 128), 
            nn.ReLU(), 
            nn.Linear(128, 128), 
            nn.ReLU(), 
            nn.Linear(128, 64), 
            nn.ReLU(), 
            nn.Linear(64, features_dim),
            ])

    def forward(self, x):
        for l in self.conv_net:
            wnds = l(wnds)
        wnds = wnds.view(wnds.size(0), -1)
        for l in self.fc_net:
            x = l(x)
        return x