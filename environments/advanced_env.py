import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

def pos_to_array(pos):
    return [pos["x"], pos["y"], pos["z"]]

class AdvancedEnv(gym.Env):
    '''
    Crée un environnement 2D modélisant une version simplifiée du problème
    '''
    def __init__(self, wind, start, goal, max_steps=1000, state_mode=0, radius=1, nb_actions=3):
        super(AdvancedEnv, self).__init__()

        # Paramètres annexes de tests
        self.state_mode = state_mode

        self.wind = np.array(wind)
        self.start_pos = start
        self.goal_pos = goal

        self.max_x = 100
        self.max_y = 100
        self.max_z = 100
        self.min_x = 0
        self.min_y = 0
        self.min_z = 0

        self.radius = radius

        # Initialisation de la position de l'agent
        self.current_pos = self.start_pos

        # Définition de constantes utiles 
        self.env_shape = self.wind.shape
        self.num_rows, self.num_cols = self.env_shape

        # Paramètres pour éviter des épisodes infinis
        self.count = 0
        self.max_steps = max_steps

        # Encodage des actions
        # x = n/2: Immobile, x > n/2: Monter de x - n/2, x < n/2: Descendre de n/2 - x
        self.nb_actions = nb_actions
        self.action_space = spaces.Discrete(n=self.nb_actions)

        # Affichage
        self.cell_size = 10
        self.screen = None

    def reset(self):
        '''
        Réinitialise la position de l'agent et l'environnement (en fonction de paramètres)
        '''
        self.current_pos = self.start_pos
        self.count = 0
        return self.currentstate()
    
    def currentstate(self):
        '''
        Renvoie l'état actuel de l'agent
        En fonction des paramètres, renvoie : [position, objectif] ou [position, objectif, vents] ou [position]
        '''
        if self.state_mode == 0:
            return np.concatenate((pos_to_array(self.current_pos), pos_to_array(self.goal_pos), self.wind.flatten()))
        if self.state_mode == 1:
            return np.concatenate((pos_to_array(self.current_pos), pos_to_array(self.goal_pos)))
        return pos_to_array(self.current_pos)

    def step(self, action):
        '''
        Fait faire l'action passée en paramètre à l'agent et renvoie :
        (nouvel_etat, reward, fini, )
        '''
        self.update_pos(action)
        self.count += 1

        # Calcul du reward en fonction de la position
        done = False
        reward = self.reward(action)
        if self.count >= self.max_steps:
            done = True

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

    def update_pos(self, action):
        new_pos = self.current_pos
        mvt = action - self.nb_actions//2
        new_pos["z"] = max(min(self.max_z, new_pos["z"] + mvt), self.min_z)
        wind_x, wind_y = self.get_wind(new_pos)
        new_pos["x"] = max(self.min_x, min(self.max_x, new_pos["x"] + wind_x))
        new_pos["y"] = max(self.min_y, min(self.max_y, new_pos["y"] + wind_y))
        self.current_pos = new_pos

    def get_sq_dist_obj(self):
        x,y = self.current_pos["x"],self.current_pos["y"]
        x_goal,y_goal = self.goal_pos["x"],self.goal_pos["y"]
        return (x-x_goal)**2 + (y-y_goal)**2

    def reward(self, action):
        dist_sq = self.get_sq_dist_obj()
        r_sq = self.radius**2
        if dist_sq <= r_sq:
            return 1
        return np.exp(r_sq - dist_sq)
    
    def get_wind(self, position):
        return self.wind[position]