import random
import numpy as np

class EpsGreedy():
    '''
    Crée un objet permettant de mettre en place une stratégie epsilon-greedy
    Epsilon peut éventuellement évoluer avec le temps
    '''
    def __init__(self, eps=0.1, refresh=None, act=.95):
        self.eps = eps
        self.refresh = refresh
        self.act = act
        self.steps = 0
    
    def pick_action(self, env, model):
        if not self.refresh is None:
            self.steps = self.steps+1 % self.refresh
            if self.steps == 0:
                self.eps = self.act * self.eps

        if np.random.rand() < self.eps:
            return env.action_space.sample()
        else:
            state = np.array([env.currentstate()])
            action = max_action(model.predict(state)[0])
            return action
        
def max_action(array):
    '''
    Renvoie l'action a avec probabilité d'autant plus grande que Q(s,a) est grand
    '''
    if array.sum() == 0:
        array = array+1
    return random.choices(range(len(array)), weights=array)[0]