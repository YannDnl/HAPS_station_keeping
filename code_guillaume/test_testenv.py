import gymnasium as gym
from environments.testenv import TestEnv
import numpy as np
import pygame

gym.register(
    id="TestEnvSimple-v0",
    entry_point="environments.testenv:TestEnv",
    kwargs={"wind": None}
)

x,y = 50, 15
wind = np.array([[j-(y//2) for j in range(y)] for i in range(x)])

env = gym.make('TestEnvSimple-v0', wind=wind, max_steps=1000, random_mode=1, render_mode="human")
obs = env.reset()

done = False

count = 0

scores = 0
score = 0
while count < 100:
    action = env.action_space.sample()  # Random action selection
    obs, reward, done, _ = env.step(action)
    score += reward
    if(done):
        print('Reward:', score)
        print('Count:', env.count)
        count+=1
        scores += score
        score = 0
        env.reset()

print(scores/(100))