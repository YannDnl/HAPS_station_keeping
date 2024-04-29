import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

## tab = np.array(np.where(np.array([[[0,0,1,0,1], [0,0,1,0,1], [0,0,1,0,1]], [[0,0,1,0,1], [0,0,1,0,1], [0,0,1,0,1]], [[0,0,1,0,1], [0,0,1,0,1], [0,0,1,0,1]]]) == 1))
## d,n = tab.shape
## print([[tab[i,j] for i in range(d)] for j in range(n)])

def rand_vec(maxs, mins=None):
    if mins == None:
        mins = np.zeros(len(maxs))
    return np.array([np.randint(mins[i], maxs[i]) for i in len(maxs)])

class TestEnv(gym.Env):
    '''
    CrÃ©e un environnement 2D modÃ©lisant une version simplifiÃ©e du problÃ¨me
    '''
    def __init__(self, wind, max_steps=200, random_mode=1, max_dev=1, radius=3, render_mode=None):
        super(TestEnv, self).__init__()

        # ParamÃ¨tres annexes de tests
        self.random_mode = random_mode
        self.render_mode = render_mode
        self.max_dev = max_dev
        
        self.wind = np.array(wind)
        x, y = self.wind.shape

        # Initialisation de la position de l'agent
        self.radius = radius

        # DÃ©finition de constantes utiles 
        self.max_vect = np.array([self.wind.shape[0]-1, self.wind.shape[1]-1])

        self.env_shape = (1,x,y)

        # ParamÃ¨tres pour Ã©viter des Ã©pisodes infinis
        self.count = 0
        self.max_steps = max_steps

        # Encodage des actions
        self.action_space = spaces.Discrete(n=3, start=0)
        self.observation_space = spaces.Dict({"wind": spaces.Box(high=(self.max_dev+1)*np.ones(self.env_shape), low=-self.max_dev*np.ones(self.env_shape), shape=self.env_shape), "pos" : spaces.Box(low=0, high=self.max_vect, shape=self.max_vect.shape), "obj" : spaces.Box(low=0, high=self.max_vect, shape=self.max_vect.shape)})

        # Affichage
        self.cell_size = 10
        self.screen = None

    def reset(self, **kwargs):
        '''
        RÃ©initialise la position de l'agent et l'environnement (en fonction de paramÃ¨tres)
        '''
        if self.random_mode == 1:
            r,c = self.wind.shape
            self.wind = np.random.randint(low=-self.max_dev, high=self.max_dev+1, size=self.wind.shape)
            self.current_pos = [np.random.randint(low=0,high=r), np.random.randint(low=0,high=c)]
            self.goal_pos = [np.random.randint(low=0,high=r), np.random.randint(low=0,high=c)]
        else:
            r,c = self.wind.shape
            self.current_pos = [np.random.randint(low=0,high=r), np.random.randint(low=0,high=c)]
            self.goal_pos = [np.random.randint(low=0,high=r), np.random.randint(low=0,high=c)]
        self.count = 0
        return self.currentstate(), {}
    
    def currentstate(self):
        '''
        Renvoie l'Ã©tat actuel de l'agent
        En fonction des paramÃ¨tres, renvoie : [position, objectif] ou [position, objectif, vents] ou [position]
        '''
        return {"wind" : self.wind.reshape(self.env_shape), "pos" : self.current_pos, "obj" : self.goal_pos}

    def step(self, action):
        '''
        Fait faire l'action passÃ©e en paramÃ¨tre Ã  l'agent et renvoie :
        (nouvel_etat, reward, fini, )
        '''
        # Traduction de l'action en mouvement vertical
        mvt = np.array([self.get_wind(), action - 1])
        
        # Calcul de la nouvelle position verticale

        self.current_pos = np.maximum(np.minimum(self.current_pos + mvt, self.max_vect), np.zeros((2)))

        self.count += 1

        done = self.count >= self.max_steps
        # Calcul du reward en fonction de la position
        reward = self.reward(action)

        self.render(self.render_mode)
        if done :
            pygame.quit()
            self.screen = None

        return self.currentstate(), reward, done, False, {}
    
    def render(self, mode=None):
        '''
        Fonction d'affichage
        '''
        if mode == None:
            return
        
        if self.screen == None:
            pygame.init()
            x, y = self.wind.shape
            self.screen = pygame.display.set_mode((x*self.cell_size, y*self.cell_size))

        self.screen.fill((255, 255, 255))  

        mxwd = np.max(np.abs(self.wind))
        for x, col in enumerate(self.wind):
            for y, wind in enumerate(col):
                cell_left = x * self.cell_size
                cell_top = y * self.cell_size
                if wind > 0:
                    pygame.draw.rect(self.screen, (0, 255*wind/mxwd, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                elif wind < 0:
                    pygame.draw.rect(self.screen, (-255*wind/mxwd, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                else:
                    pygame.draw.rect(self.screen, (0, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                
        pygame.draw.circle(self.screen, (255, 255, 0), (float(self.goal_pos[0])*self.cell_size + self.cell_size/2, float(self.goal_pos[1])*self.cell_size + self.cell_size/2), 5)
        pygame.draw.circle(self.screen, (0, 0, 255), (float(self.current_pos[0])*self.cell_size + self.cell_size/2, float(self.current_pos[1])*self.cell_size + self.cell_size/2), 5)

        pygame.display.update()
        if mode != None:
            pygame.time.wait(1)

    def reward(self, action):
        dist_sq = np.sum((self.current_pos-self.goal_pos)**2)
        r_sq = self.radius**2
        if dist_sq <= r_sq:
            return 1
        return np.exp(r_sq - dist_sq)
    
    def get_wind(self, pos=None):
        if pos == None:
            return self.wind[int(self.current_pos[0]), int(self.current_pos[1])]
        return self.wind[int(pos[0]), int(pos[1])]