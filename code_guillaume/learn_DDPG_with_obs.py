import torch
import gymnasium as gym
from models_DDPG import CustomFeatureExtractor2D, CustomFeatureExtractor

import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.ddpg import MultiInputPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise

from utils_DDPG import evaluate, load_best_weights, visual_test

from tqdm import tqdm

device=torch.device("cuda")

gym.register(
    id="TestEnvContMultiagent2DWithObservation-v0",
    entry_point="environments.2D_env_contin_wt_obs_multiagent:TestEnv",
    kwargs={"wind": None}
)
gym.register(
    id="TestEnvContMultiagentWithObservation-v0",
    entry_point="environments.env_contin_wt_obs_multiagent:TestEnv",
    kwargs={"wind": None}
)

x,y,z= 50, 50, 50
x,y,z= 20, 20, 20
wind = torch.rand((2,x,y,z))

n_agents=10
env = gym.make('TestEnvContMultiagentWithObservation-v0', wind=wind, max_steps=1000, random_mode=0, n_agents=n_agents, render_mode=None)
env = DummyVecEnv([lambda: env])

# Add noise
noise_stddev = 10
#action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(env.action_space.shape), sigma=noise_stddev * np.ones(env.action_space.shape))
action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape), sigma=noise_stddev * np.ones(env.action_space.shape))

# Define custom archs

policy_kwargs = dict(features_extractor_class=CustomFeatureExtractor, features_extractor_kwargs=dict(features_dim=128))

# Create the DDPG agent
model = DDPG('MultiInputPolicy', env, policy_kwargs=policy_kwargs, verbose=1, action_noise=action_noise, buffer_size=100000, device=device, optimize_memory_usage=True, batch_size=512)

load_best_weights(model,f"ddpg_multiagent_3d_wt_obs/{n_agents}")

# Training
for i in tqdm(range(100)):
    load_best_weights(model,f"ddpg_multiagent_3d_wt_obs/{n_agents}")
    model.learn(total_timesteps=50000, progress_bar=True)
    # Save the trained model
    score = evaluate(model, 100)
    print(score)
    #visual_test(model)
    model.save(f"ddpg_multiagent_3d_wt_obs/{n_agents}/ddpg_{score}")                                                    
# Close the environment
env.close()