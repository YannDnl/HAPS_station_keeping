import numpy as np
import gym
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise

from utils_DDPG import evaluate, load_best_weights, visual_test

from tqdm import tqdm

# Create and wrap the environment
gym.register(
    id="TestEnvCont-v0",
    entry_point="environments.env_contin:TestEnv",
    kwargs={"wind": None}
)

h,w = 11, 50
wind = np.array([[i-(h//2) for j in range(w)] for i in range(h)])

env = gym.make('TestEnvCont-v0', wind=wind, start=[0,w-1], goal=[h-1,w-1], max_steps=200, random_mode=1, state_mode=1, render_mode=None)
env = DummyVecEnv([lambda: env])


# Add noise
noise_stddev = .01
#action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(env.action_space.shape), sigma=noise_stddev * np.ones(env.action_space.shape))
action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape), sigma=noise_stddev * np.ones(env.action_space.shape))

# Create the DDPG agent
model = DDPG('MlpPolicy', env, verbose=1, action_noise=action_noise)

load_best_weights(model)

#visual_test(model)

# Train the agent
for i in tqdm(range(100)):
    model.learn(total_timesteps=50000)
    # Save the trained model
    score = evaluate(model, 100)
    print(score)
    #visual_test(model)
    model.save(f"ddpg_wghts/ddpg_{score}")

# Close the environment
env.close()