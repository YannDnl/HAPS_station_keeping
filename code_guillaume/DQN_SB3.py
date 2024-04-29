import gymnasium as gym
import numpy as np
import torch 

from tqdm import tqdm

from stable_baselines3 import DQN
from stable_baselines3.dqn import MultiInputPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from environments.testenv import TestEnv
from utils_DQN import evaluate
from models_DQN import CustomFeatureExtractor2D


gym.register(
    id="TestEnvSimple-v0",
    entry_point="environments.testenv:TestEnv",
    kwargs={"wind": None}
)

device = torch.device("cuda")

x,y = 50, 30
wind = np.array([[j-(y//2) for j in range(y)] for i in range(x)])

env = gym.make('TestEnvSimple-v0', wind=wind, max_steps=1000, random_mode=1, render_mode=None)
# env = TestEnv(wind=wind, max_steps=1000, random_mode=1, render_mode="human")
# env = DummyVecEnv([lambda: env])

model = DQN(MultiInputPolicy, policy_kwargs={"net_arch":[128, 64, 64, 32], "features_extractor_class":CustomFeatureExtractor2D}, env=env, device=device, verbose=1, batch_size=256)

print(evaluate(model, 100))
# # Training
for i in tqdm(range(100)):
    model.learn(total_timesteps=50000, progress_bar=True)
    # Save the trained model
    score = evaluate(model, 100)
    print(score)
    #visual_test(model)
    model.save(f"dqn_sb3_single/dqn_{score}")

env.close()