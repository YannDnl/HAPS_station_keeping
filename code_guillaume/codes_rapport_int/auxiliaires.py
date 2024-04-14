import os
from keras.layers import Flatten, Dense, InputLayer
from keras.models import Sequential
from keras.optimizers.legacy import Adam
from keras.losses import MeanSquaredError
import random as rd

h,w = 10, 50

def create_model(input_shape=2+2+h*w, num_actions=3, summary=False):
    '''
    Crée un NN à utiliser pour approcher Q(s,a)
    '''

class EpsGreedy():
    def __init__(self, eps=0.1, refresh=None, act=.95):
        '''
        Crée un objet permettant de mettre en place une stratégie epsilon-greedy
        Epsilon peut éventuellement évoluer avec le temps
        '''
    
    def pick_action(self, env, model):
        '''
        Renvoie l’action choisie en fonction de la stratégie
        '''
        
def max_action(array):
    '''
    Renvoie l'action a avec probabilité d'autant plus grande que Q(s,a) est grand
    '''