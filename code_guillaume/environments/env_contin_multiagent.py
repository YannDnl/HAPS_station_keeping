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
    def __init__(self, wind, start, goal, max_steps=200, random_mode=0, max_dev=1, state_mode=1, radius=1, render_mode=None, n_agents=1):
        super(TestEnv, self).__init__()

        # Paramètres annexes de tests
        self.render_mode = render_mode
        self.random_mode = random_mode
        self.state_mode = state_mode
        self.max_dev = max_dev
        if self.random_mode == 2:
            r,c = wind.shape
            self.wind = np.random.randint(low=-self.max_dev, high=self.max_dev+1, size=wind.shape)
            self.start_pos = np.array([np.random.uniform(low=0,high=c), np.random.uniform(low=0,high=r)])
            self.goal_pos = np.array([np.random.uniform(low=0,high=c), np.random.uniform(low=0,high=r)])
        elif self.random_mode == 1:
            r,c = wind.shape
            self.wind = np.array(wind)
            self.start_pos = np.array([np.random.uniform(low=0,high=c), np.random.uniform(low=0,high=r)])
            self.goal_pos = np.array([np.random.uniform(low=0,high=c), np.random.uniform(low=0,high=r)])
        else:
            self.wind = np.array(wind)
            self.start_pos = np.array(start)
            self.goal_pos = np.array(goal)

        # Initialisation de la position de l'agent
        self.n_agents = n_agents

        self.current_pos = [self.start_pos for i in range(self.n_agents)]
        self.radius = radius

        # Définition de constantes utiles 
        self.env_shape = self.wind.shape
        self.num_rows, self.num_cols = self.env_shape

        # Paramètres pour éviter des épisodes infinis
        self.count = 0
        self.max_steps = max_steps

        # Encodage des actions
        # x = n/2: Immobile, x > n/2: Monter de x - n/2, x < n/2: Descendre de n/2 - x
        #self.action_space = spaces.Box(-1*self.num_rows, 1*self.num_rows, (1,))
        self.action_space = spaces.Box(-1, 1, (1*self.n_agents,))
        self.observation_space = spaces.Box(low=0, high=self.num_rows, shape=(2,self.n_agents + 1))

        # Affichage
        self.cell_size = 10
        self.screen = None

    def reset(self, seed=None, *args):
        '''
        Réinitialise la position de l'agent et l'environnement (en fonction de paramètres)
        '''
        if self.random_mode == 2:
            r,c = self.wind.shape
            self.wind = np.random.randint(low=-self.max_dev, high=self.max_dev+1, size=self.wind.shape)
            self.start_pos = np.array([np.random.uniform(low=0,high=c), np.random.uniform(low=0,high=r)])
            self.goal_pos = np.array([np.random.uniform(low=0,high=c), np.random.uniform(low=0,high=r)])
        elif self.random_mode == 1:
            r,c = self.wind.shape
            self.start_pos = np.array([np.random.uniform(low=0,high=c), np.random.uniform(low=0,high=r)])
            self.goal_pos = np.array([np.random.uniform(low=0,high=c), np.random.uniform(low=0,high=r)])
        
        self.current_pos = np.array([self.start_pos for i in range(self.n_agents)])
        self.count = 0
        return self.currentstate(), {}
    
    def currentstate(self):
        '''
        Renvoie l'état actuel de l'agent
        En fonction des paramètres, renvoie : [position, objectif] ou [position, objectif, vents] ou [position]
        '''
        if self.state_mode == 0:
            return [np.concatenate((self.current_pos, self.goal_pos)), self.wind]
        if self.state_mode == 1:
            return np.concatenate((self.current_pos, self.goal_pos.reshape((1,2)))).T
        return self.current_pos

    def step(self, action, *args):
        '''
        Fait faire l'action passée en paramètre à l'agent et renvoie :
        (nouvel_etat, reward, fini, )
        '''
        new_pos = np.array(self.current_pos)

        mvt = np.matrix([self.get_wind(),action.reshape((self.n_agents,))]).T
        maxs = np.matrix([(self.num_cols - .1)*np.ones(self.n_agents), (self.num_rows - .1)*np.ones(self.n_agents)]).T
        mins = np.zeros((self.n_agents, 2))

        # Calcul de la nouvelle position verticale
        new_pos = np.maximum(np.minimum(new_pos + mvt, maxs), mins)

        self.current_pos = new_pos

        self.count += 1

        done = self.count >= self.max_steps
        # Calcul du reward en fonction de la position
        reward = self.reward(action)

        self.render(self.render_mode)
        
        if done :
            pygame.quit()
            self.screen = None
        
        return self.currentstate(), reward, done, False, {}
    
    def render(self, mode=None, *args):
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
                
        pygame.draw.circle(self.screen, (0, 255, 255), (self.start_pos[0]*self.cell_size, self.start_pos[1]*self.cell_size), 5)
        pygame.draw.circle(self.screen, (255, 255, 0), (self.goal_pos[0]*self.cell_size, self.goal_pos[1]*self.cell_size), 5)
        for b in self.current_pos:
            x,y = b[0,0], b[0,1]
            pygame.draw.circle(self.screen, (0, 0, 255), (x*self.cell_size, y*self.cell_size), 5)

        pygame.display.update()
        if mode != None:
            pygame.time.wait(1)

    def reward(self, action):
        dist_sq = min([np.sum(np.square(pos-self.goal_pos)) for pos in self.current_pos])
        act_pty = np.sum(np.square(action))
        r_sq = self.radius**2
        if dist_sq <= r_sq:
            return 1/(1+act_pty)
        return 1/(dist_sq*(1+act_pty)) #np.exp(r_sq - dist_sq)
    
    def get_dist(self, goal, i):
        return np.sum(np.square(self.current_pos[i]-np.array(goal)))

    def get_wind(self):
        winds = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            x,y = self.current_pos[i, 0], self.current_pos[i, 1]

            d_ul = self.get_dist([int(x), int(y)], i)
            d_ur = self.get_dist([int(x) + 1, int(y)], i)
            d_ll = self.get_dist([int(x), int(y) + 1], i)
            d_lr = self.get_dist([int(x) + 1, int(y) + 1], i)

            wind = d_ul*self.wind[int(y), int(x)]
            dist_tot = d_ul
            if int(x) < self.num_cols - 1:
                wind += d_ur*self.wind[int(y), int(x) + 1]
                dist_tot += d_ur
            if int(y) < self.num_rows - 1:
                wind += d_ll*self.wind[int(y) + 1, int(x)]
                dist_tot += d_ll
            if int(x) < self.num_cols - 1 and int(y) < self.num_rows - 1:
                wind += d_lr*self.wind[int(y) + 1, int(x) + 1]
                dist_tot += d_lr
            
            winds[i] = wind/dist_tot

        return winds