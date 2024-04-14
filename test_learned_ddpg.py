import os
import numpy as np
import torch
import gym
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise

from models_DDPG import CustomFeatureExtractor2D
from utils_DDPG import load_best_weights, evaluate

# Create and wrap the environment
gym.register(
    id="TestEnvCont-v0",
    entry_point="environments.env_contin:TestEnv",
    kwargs={"wind": None}
)
gym.register(
    id="TestEnvContMultiagent-v0",
    entry_point="environments.env_contin_multiagent:TestEnv",
    kwargs={"wind": None}
)
gym.register(
    id="TestEnvContMultiagent2DWithObservation-v0",
    entry_point="environments.2D_env_contin_wt_obs_multiagent:TestEnv",
    kwargs={"wind": None}
)

h,w = 50, 11
# wind = np.array([[i-(h//2) for j in range(w)] for i in range(h)])
wind = torch.rand((1,h,w))

# env = gym.make('TestEnvCont-v0', wind=wind, start=[0,w-1], goal=[h-1,w-1], max_steps=200, random_mode=1, state_mode=1, render_mode="human")

n_agents = 10
env = gym.make('TestEnvContMultiagent2DWithObservation-v0', wind=wind, max_steps=200, random_mode=0, n_agents=n_agents, render_mode="human")
env = DummyVecEnv([lambda: env])


# Add noise
noise_stddev = 10
#action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(env.action_space.shape), sigma=noise_stddev * np.ones(env.action_space.shape))
action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape), sigma=noise_stddev * np.ones(env.action_space.shape))

policy_kwargs = dict(features_extractor_class=CustomFeatureExtractor2D, features_extractor_kwargs=dict(features_dim=128))

# Create the DDPG agent
model = DDPG('MultiInputPolicy', env, policy_kwargs=policy_kwargs, verbose=1, action_noise=action_noise)

load_best_weights(model=model, dir=f"ddpg_multiagent_wt_obs/{n_agents}")

# Train the agent
print(evaluate(model=model, n=100))

# Close the environment
env.close()