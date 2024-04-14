import os
import numpy as np
from tqdm import tqdm
from torch import Tensor

def load_best_weights(model, dir="ddpg_wghts"):
    chkps_dir = os.path.join(os.curdir, dir)
    weights = os.listdir(chkps_dir)
    if len(weights) > 0:
        scores = [float(wght.split("_")[-1].split(".")[0] + "." + wght.split("_")[-1].split(".")[1]) for wght in weights]
        argmax = scores.index(max(scores))
        model.set_parameters(os.path.join(chkps_dir, weights[argmax]))

def evaluate(model, n=10):
    ep_rwds = []
    for i in tqdm(range(n)):
        state = model.env.reset()
        done = False
        episode_rwd = 0
        while not done:
            action = model.predict(observation=state)
            #print(model.actor(Tensor(state)), action)
            state, reward, done, _ = model.env.step(action)
            episode_rwd += reward
        ep_rwds.append(episode_rwd)
    return np.mean(ep_rwds)

def visual_test(model, n=10):
    for i in range(n):
        state = model.env.reset()
        done = False
        while not done:
            action = model.predict(observation=state)
            state, reward, done, _ = model.env.step(action)
            print(model.env.render(mode="human"))