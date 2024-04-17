import gymnasium as gym
from gymnasium import spaces
import torch
import pygame

## tab = np.array(np.where(np.array([[[0,0,1,0,1], [0,0,1,0,1], [0,0,1,0,1]], [[0,0,1,0,1], [0,0,1,0,1], [0,0,1,0,1]], [[0,0,1,0,1], [0,0,1,0,1], [0,0,1,0,1]]]) == 1))
## d,n = tab.shape
## print([[tab[i,j] for i in range(d)] for j in range(n)])

class TestEnv(gym.Env):
    '''
    Crée un environnement 2D modélisant une version simplifiée du problème
    '''
    def __init__(self, wind, start=[0,0], goal=[0,0], max_steps=200, random_mode=0, radius=1, render_mode=None, n_agents=1):
        super(TestEnv, self).__init__()

        # Paramètres annexes de tests
        self.render_mode = render_mode
        self.random_mode = random_mode

        self.n_agents = n_agents
        self.wind_shape = wind.shape
        self.bounds = torch.tensor(self.wind_shape[1:]).view((2,1))
        self.radius = radius

        self.max_wind= 5
        self.max_mvt = 1

        self.reset()

        # Paramètres pour éviter des épisodes infinis
        self.count = 0
        self.max_steps = max_steps

        # Encodage des actions
        # x = n/2: Immobile, x > n/2: Monter de x - n/2, x < n/2: Descendre de n/2 - x
        #self.action_space = spaces.Box(-1*self.num_rows, 1*self.num_rows, (1,))
        self.action_space = spaces.Box(-self.max_mvt, self.max_mvt, (1*self.n_agents,))
        self.observation_space = spaces.Dict({'wind': spaces.Box(low=-self.max_wind, high=self.max_wind, shape=self.wind_shape), 'pos': spaces.Box(low=0, high=float(max(self.bounds)), shape=(2, self.n_agents + 1))})

        # Affichage
        self.cell_size = 10
        self.screen = None

    def reset(self, seed=None, *args):
        '''
        Réinitialise la position de l'agent et l'environnement (en fonction de paramètres)
        '''
        if self.random_mode == 1:
            self.wind = 2*self.max_wind*(torch.rand(self.wind_shape) - .5)
            # self.wind = torch.zeros(self.wind_shape)
            self.goal_pos = torch.mul(self.bounds, torch.rand((2, 1)))
            self.current_pos = torch.mul(self.bounds.expand((2, self.n_agents)), torch.rand((2, self.n_agents)))
        else:
            self.wind = self.max_wind*(torch.randn(self.wind_shape))
            self.goal_pos = torch.mul(self.bounds, torch.rand((2, 1)))
            self.current_pos = torch.mul(self.bounds.expand((2, self.n_agents)), torch.rand((2, self.n_agents)))

        self.count = 0
        return self.currentstate(), {}
    
    def currentstate(self):
        '''
        Renvoie l'état actuel de l'agent
        En fonction des paramètres, renvoie : [position, objectif] ou [position, objectif, vents] ou [position]
        '''
        return {'wind': self.wind, 'pos': torch.cat((self.goal_pos, self.current_pos), dim=-1)}

    def step(self, action, *args):
        '''
        Fait faire l'action passée en paramètre à l'agent et renvoie :
        (nouvel_etat, reward, fini, )
        '''

        action = torch.tensor(action)
        mvt = torch.cat([self.get_wind(), action.view((1,self.n_agents))], dim=0)
        maxs = self.bounds.expand((2, self.n_agents))
        mins = torch.zeros((2, self.n_agents))

        # Calcul de la nouvelle position verticale
        self.current_pos = torch.maximum(torch.minimum(self.current_pos + mvt, maxs - .1), mins)

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
            x, y = self.bounds.view((2))
            self.screen = pygame.display.set_mode((x * self.cell_size, y * self.cell_size))

        self.screen.fill((255, 255, 255))  

        for wind_dim in self.wind:
            for x, x_vect in enumerate(wind_dim):
                for y, wind in enumerate(x_vect):
                    cell_left = x * self.cell_size
                    cell_top = y * self.cell_size
                    if wind > 0:
                        pygame.draw.rect(self.screen, (0, 255*wind/(torch.max(torch.abs(self.wind))), 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                    elif wind < 0:
                        pygame.draw.rect(self.screen, (-255*wind/(torch.max(torch.abs(self.wind))), 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                    else:
                        pygame.draw.rect(self.screen, (0, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                
        pygame.draw.circle(self.screen, (255, 255, 0), (float(self.goal_pos[0])*self.cell_size, float(self.goal_pos[1])*self.cell_size), 5)
        for b in self.current_pos.T:
            x,y = b
            pygame.draw.circle(self.screen, (0, 0, 255), (float(x)*self.cell_size, float(y)*self.cell_size), 5)

        pygame.display.update()
        if mode != None:
            pygame.time.wait(1)

    def reward(self, action):
        dist_sq = min([torch.sum(torch.square(pos-self.goal_pos)) for pos in self.current_pos])
        act_pty = torch.sum(torch.square(action))
        r_sq = self.radius**2
        if dist_sq <= r_sq:
            return 1/(1+act_pty)
        return 1/(dist_sq*(1+act_pty)) #np.exp(r_sq - dist_sq)
    
    def get_dist(self, pos, goal):
        return torch.sum(torch.square(torch.tensor(pos)-torch.tensor(goal)))

    def get_wind(self):
        winds = torch.zeros((1, self.n_agents))
        for i, ac in enumerate(self.current_pos.T):
            x,y = ac

            int_pts = torch.tensor([[int(x), int(y)], [int(x) + 1, int(y)], [int(x), int(y) + 1], [int(x) + 1, int(y) + 1]])

            wind_x = 0
            dist_tot = 0

            for p in int_pts:
                if torch.min(p < self.bounds.T):
                    d = self.get_dist(ac, p)
                    x_p, y_p = p
                    wind_x += d*self.wind[0, x_p, y_p]
                    dist_tot += d
            
            winds[0, i] = wind_x/dist_tot

        return winds