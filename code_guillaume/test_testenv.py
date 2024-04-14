import gymnasium as gym
from environments.testenv import TestEnv
import numpy as np
import pygame

gym.register(
    id="TestEnvSimple-v0",
    entry_point="testenv:TestEnv",
    kwargs={"wind": None}
)

h,w = 5, 10
wind = np.array([[i-(h//2) for j in range(w)] for i in range(h)])

env = gym.make('TestEnvSimple-v0', wind=wind, start=[0,w-1], goal=[h-1,w-1], max_steps=1000, random_mode=0)
obs = env.reset()
env.render()

done = False

count = 0

while count < 10:
    action = env.action_space.sample()  # Random action selection
    obs, reward, done, _ = env.step(action)
    env.render(mode="human")
    if(done):
        print('Reward:', reward)
        print('Count:', env.count)
        count+=1
        env.reset()