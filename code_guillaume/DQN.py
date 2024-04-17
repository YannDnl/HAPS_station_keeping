import numpy as np
from pygame.locals import *
import tensorflow as tf

from keras import __version__
tf.keras.__version = __version__

from rl.agents import DQNAgent
#from testenv import TestEnv

from environments.test_env_raw import TestEnv

from DQNSetup import DQNSetup
from model import create_model, load_best_weights, load_random_weights
from policies import EpsGreedy


# Définition d'un modèle de tests
h,w = 11, 50
wind = np.array([[i-(h//2) for j in range(w)] for i in range(h)])
start=[0,w-1]
goal=[(h-1),(w-1)]
#input_shape = (1, 2+2+h*w)
input_shape = 2+2

# Création de l'environnement
env = TestEnv(wind=wind, start=start, goal=goal, max_steps=200, random_mode=1, state_mode=1)

# Création du NN
model = create_model(input_shape=input_shape, summary=True, num_actions=env.env_shape[0])
load_best_weights(model)
#model = load_random_weights(model)

# Définition de la policy pour l'apprentissage RL (pour le leverage exploration/exploitation)
epsgreedy = EpsGreedy(.1, refresh = None)

# Mise en place de l'environnement de DQN
DQN = DQNSetup(env, model, epsgreedy.pick_action, gamma=.99, alpha=1, batch_size=32, buffer_limit=10000)


# Entraînement et tests

for i in range(50):
    print(i)
    #load_best_weights(DQN.model)
    #epsgreedy.eps=.9
    DQN.fit(nb_episodes=100, visualize=None, verbose=2)
    DQN.test(nb_episodes=10, visualize="human")
    DQN.export_model(n=1000)