import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

## tab = np.array(np.where(np.array([[[0,0,1,0,1], [0,0,1,0,1], [0,0,1,0,1]], [[0,0,1,0,1], [0,0,1,0,1], [0,0,1,0,1]], [[0,0,1,0,1], [0,0,1,0,1], [0,0,1,0,1]]]) == 1))
## d,n = tab.shape
## print([[tab[i,j] for i in range(d)] for j in range(n)])

class TestEnv(gym.Env):
    '''
    Crée un environnement 2D modélisant une version simplifiée du problème
    '''
    def __init__(self, wind, start, goal, max_steps=200, random_mode=0, max_dev=1, state_mode=0, radius=1):
        super(TestEnv, self).__init__()

        # Paramètres annexes de tests
        self.random_mode = random_mode
        self.state_mode = state_mode
        self.max_dev = max_dev
        if self.random_mode == 2:
            r,c = wind.shape
            self.wind = np.random.randint(low=-self.max_dev, high=self.max_dev+1, size=wind.shape)
            self.start_pos = [np.random.randint(low=0,high=r), np.random.randint(low=0,high=c)]
            self.goal_pos = [np.random.randint(low=0,high=r), np.random.randint(low=0,high=c)]
        elif self.random_mode == 1:
            r,c = wind.shape
            self.wind = np.array(wind)
            self.start_pos = [np.random.randint(low=0,high=r), np.random.randint(low=0,high=c)]
            self.goal_pos = [np.random.randint(low=0,high=r), np.random.randint(low=0,high=c)]
        else:
            self.wind = np.array(wind)
            self.start_pos = start
            self.goal_pos = goal

        # Initialisation de la position de l'agent
        self.current_pos = self.start_pos
        self.radius = radius

        # Définition de constantes utiles 
        self.env_shape = self.wind.shape
        self.num_rows, self.num_cols = self.env_shape

        # Paramètres pour éviter des épisodes infinis
        self.count = 0
        self.max_steps = max_steps

        # Encodage des actions
        # x = n/2: Immobile, x > n/2: Monter de x - n/2, x < n/2: Descendre de n/2 - x
        self.action_space = spaces.Discrete(n=self.env_shape[0])
        self.observation_space = spaces.MultiDiscrete(self.env_shape)

        # Affichage
        self.cell_size = 10
        self.screen = None

    def reset(self):
        '''
        Réinitialise la position de l'agent et l'environnement (en fonction de paramètres)
        '''
        if self.random_mode == 2:
            r,c = self.wind.shape
            self.wind = np.random.randint(low=-self.max_dev, high=self.max_dev+1, size=self.wind.shape)
            self.start_pos = [np.random.randint(low=0,high=r), np.random.randint(low=0,high=c)]
            self.goal_pos = [np.random.randint(low=0,high=r), np.random.randint(low=0,high=c)]
        elif self.random_mode == 1:
            r,c = self.wind.shape
            self.start_pos = [np.random.randint(low=0,high=r), np.random.randint(low=0,high=c)]
            self.goal_pos = [np.random.randint(low=0,high=r), np.random.randint(low=0,high=c)]
        self.current_pos = self.start_pos
        self.count = 0
        return self.currentstate()
    
    def currentstate(self):
        '''
        Renvoie l'état actuel de l'agent
        En fonction des paramètres, renvoie : [position, objectif] ou [position, objectif, vents] ou [position]
        '''
        if self.state_mode == 0:
            return np.concatenate((self.current_pos, self.goal_pos, self.wind.flatten()))
        if self.state_mode == 1:
            return np.concatenate((self.current_pos, self.goal_pos))
        return self.current_pos

    def step(self, action):
        '''
        Fait faire l'action passée en paramètre à l'agent et renvoie :
        (nouvel_etat, reward, fini, )
        '''
        new_pos = np.array(self.current_pos)
        # Traduction de l'action en mouvement vertical
        mvt = action - self.env_shape[0]//2
        
        # Calcul de la nouvelle position verticale
        new_pos[0] = np.floor(max(min(self.num_rows - 1, new_pos[0] + mvt), 0))
        # Déduction de la nouvelle position horizontale
        new_pos[1] = np.floor(max(0, min(self.num_cols - 1, new_pos[1] + self.wind[self.current_pos[0],self.current_pos[1]])))

        self.current_pos = new_pos

        self.count += 1

        done = self.count >= self.max_steps
        # Calcul du reward en fonction de la position
        reward = self.reward(action)

        return self.currentstate(), reward, done, {}
    
    def render(self, mode=None):
        '''
        Fonction d'affichage
        '''
        if mode == None:
            return
        
        if self.screen == None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.num_cols * self.cell_size, self.num_rows * self.cell_size))

        self.screen.fill((255, 255, 255))  

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell_left = col * self.cell_size
                cell_top = row * self.cell_size
                wind = self.wind[row, col]
                if wind > 0:
                    pygame.draw.rect(self.screen, (0, 255*wind/(np.max(np.abs(self.wind))), 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                elif wind < 0:
                    pygame.draw.rect(self.screen, (-255*wind/(np.max(np.abs(self.wind))), 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                else:
                    pygame.draw.rect(self.screen, (0, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                
                if np.array_equal(np.array(self.current_pos), np.array([row, col])):
                    pygame.draw.rect(self.screen, (0, 0, 255), (cell_left, cell_top, self.cell_size, self.cell_size))
                if np.array_equal(np.array(self.goal_pos), np.array([row, col])):
                    pygame.draw.rect(self.screen, (0, 255, 255), (cell_left, cell_top, self.cell_size, self.cell_size))
                if np.array_equal(np.array(self.start_pos), np.array([row, col])):
                    pygame.draw.rect(self.screen, (255, 255, 0), (cell_left, cell_top, self.cell_size, self.cell_size))

        pygame.display.update()
        if mode != None:
            pygame.time.wait(1)

    def reward(self, action):
        dist_sq = np.sum((self.current_pos-self.goal_pos)**2)
        r_sq = self.radius**2
        if dist_sq <= r_sq:
            return 1
        return np.exp(r_sq - dist_sq)
