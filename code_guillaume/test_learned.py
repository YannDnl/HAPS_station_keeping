import os
import numpy as np
import gymnasium as gym
import math
from PIL import Image
import pygame, sys
from pygame.locals import *
import tensorflow as tf

from keras import __version__
tf.keras.__version = __version__

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers.legacy import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from environments.testenv import TestEnv
from model import create_model, load_best_weights, create_model_sigm

gym.register(
    id="TestEnvSimple-v0",
    entry_point="testenv:TestEnv",
    kwargs={"wind": None}
)

h,w = 11, 50
#h,w = 5, 10
wind = np.array([[i-(h//2) for j in range(w)] for i in range(h)])
start=[0,w-1]
goal=[(h-1),(w-1)]

r,c = h,w
max_dev = 1
# wind = np.random.randint(low=-max_dev, high=max_dev+1, size=(r,c))
# start = [np.random.randint(low=0,high=r), np.random.randint(low=0,high=c)]
# goal = [np.random.randint(low=0,high=r), np.random.randint(low=0,high=c)]

env = gym.make('TestEnvSimple-v0', wind=wind, start=start, goal=goal, max_steps=200, random_mode=1, state_mode=1)

input_shape = env.observation_space.shape[0]
num_actions = env.action_space.n

model = create_model(input_shape=(1,2+2), summary=True, num_actions=env.env_shape[0])
load_best_weights(model, "chckpt")

agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000, window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=num_actions,
    nb_steps_warmup=10,
    target_model_update=0.01
)


agent.compile(Adam(learning_rate=0.001), metrics=["mae"])

results = agent.test(env, nb_episodes=10, visualize="human")

mean_res = np.mean(results.history["episode_reward"])
print(mean_res)

env.close()