import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class TestEnv(gym.Env):
    def __init__(self, wind, max_steps=None):
        '''
        Crée un environnement 2D modélisant une version simplifiée du problème
        wind est la carte des vents
        Les positions de départ et d'objectif sont randomisées
        max_steps est un nombre maximal de pas par épisode, pour éviter des épisodes infinis
        '''
        # Encodage des actions
        # x = n/2: Immobile, x > n/2: Monter de x - n/2, x < n/2: Descendre de n/2 - x
        self.action_space = spaces.Discrete(n=self.env_shape[0])
        self.observation_space = spaces.MultiDiscrete(self.env_shape)

    def reset(self):
        '''
        Réinitialise la position de l'agent et l'environnement
        Renvoie le nouvel état
        '''
    
    def currentstate(self):
        '''
        Renvoie l'état actuel de l'agent
        En fonction des paramètres, renvoie : [position, objectif] ou [position, objectif, vents] ou [position]
        '''

    def step(self, action):
        '''
        Fait faire l'action passée en paramètre à l'agent et renvoie :
        (nouvel_etat, reward, fini, )
        '''
        # Mise à jour de la position actuelle
        self.current_state = self.update_position(self.current_state, action)

        # Calcul du reward en fonction de la position
        if np.array_equal(self.current_pos, self.goal_pos):
            reward = 100
            done = True
        elif self.max_steps != None:
            reward = 1/np.sum((self.current_pos-self.goal_pos)**2)
            if self.count >= self.max_steps:
                done = True
            else:
                done = False
        else:
            reward = 1/np.sum((self.current_pos-self.goal_pos)**2)
            done = False

        return self.currentstate(), reward, done, {}
    
    def render(self, mode=None):
        '''
        Fonction d'affichage
        '''

    def update_position(self):
        return