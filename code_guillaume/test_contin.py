import torch
import gymnasium as gym
import numpy as np
import pygame
from models_DDPG import CustomActorNetwork2D, CustomCriticNetwork2D

gym.register(
    id="TestEnvContMultiagentWithObservation-v0",
    entry_point="environments.env_contin_wt_obs_multiagent:TestEnv",
    kwargs={"wind": None}
)

h,w = 11, 50
wind = torch.rand((2,h,w,50))

env = gym.make('TestEnvContMultiagentWithObservation-v0', wind=wind, max_steps=200, random_mode=1, n_agents=10, render_mode="human")
env.reset()

# act = CustomActorNetwork2D(env.observation_space, env.action_space)
# cri = CustomCriticNetwork2D(env.observation_space, env.action_space)

done = False

count = 0

while count < 100:
    action = env.action_space.sample()  # Random action selection
    obs, reward, done, __, _ = env.step(action)
    # env.render(mode="human")
    if(done):
        print('Reward:', reward)
        print('Count:', env.count)
        count+=1
        env.reset()