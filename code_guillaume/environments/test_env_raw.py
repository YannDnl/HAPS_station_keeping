import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

## tab = np.array(np.where(np.array([[[0,0,1,0,1], [0,0,1,0,1], [0,0,1,0,1]], [[0,0,1,0,1], [0,0,1,0,1], [0,0,1,0,1]], [[0,0,1,0,1], [0,0,1,0,1], [0,0,1,0,1]]]) == 1))
## d,n = tab.shape
## print([[tab[i,j] for i in range(d)] for j in range(n)])

class TestEnv():
    def __init__(self, wind, start, goal, max_steps=None, random_mode=0, max_dev=1, state_mode=0):

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

        self.current_pos = self.start_pos
        self.env_shape = self.wind.shape
        self.num_rows, self.num_cols = self.env_shape

        self.count = 0
        self.max_steps = max_steps

        # 0: None, 1: Up, 2: Down
        self.action_space = spaces.Discrete(n=self.env_shape[0])

        self.observation_space = spaces.MultiDiscrete(self.env_shape)

        self.cell_size = 10
        self.screen = None

#        pygame.init()
#        self.cell_size = 10
#        self.screen = pygame.display.set_mode((self.num_cols * self.cell_size, self.num_rows * self.cell_size))

    def reset(self):
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
        if self.state_mode == 0:
            return np.concatenate((self.current_pos, self.goal_pos, self.wind.flatten()))
        if self.state_mode == 1:
            return np.concatenate((self.current_pos, self.goal_pos))
        return self.current_pos

    def step(self, action):
        new_pos = np.array(self.current_pos)
        mvt = action - self.env_shape[0]//2
        #if action == 1:
        #    new_pos[0] = np.floor(max(0, new_pos[0] - 1))
        #if action == 2:
        #    new_pos[0] = np.floor(min(self.num_rows - 1, new_pos[0] + 1))
        
        new_pos[0] = np.floor(max(min(self.num_rows - 1, new_pos[0] + mvt), 0))
        new_pos[1] = np.floor(max(0, min(self.num_cols - 1, new_pos[1] + self.wind[self.current_pos[0],self.current_pos[1]])))

        self.current_pos = new_pos

        self.count += 1

        #max_steps_reward = self.max_steps
        #if self.max_steps == None :
        #    max_steps_reward = max(10000, self.count)

        if np.array_equal(self.current_pos, self.goal_pos):
            reward = 100 #self.count
            done = True
        elif self.max_steps != None:
            if self.count >= self.max_steps:
                reward = 1/np.sum((self.current_pos-self.goal_pos)**2)#(self.count**2)*(np.sum((self.current_pos-self.goal_pos)**2))
                done = True
            else:
                reward = 1/np.sum((self.current_pos-self.goal_pos)**2)
                done = False
        else:
            reward = 1/np.sum((self.current_pos-self.goal_pos)**2)
            done = False

        return self.currentstate(), reward, done, {}
    
    def render(self, mode=None):
        # Clear the screen
        if mode == None:
            return
        
        if self.screen == None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.num_cols * self.cell_size, self.num_rows * self.cell_size))

        self.screen.fill((255, 255, 255))  

        # Draw env elements one cell at a time
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell_left = col * self.cell_size
                cell_top = row * self.cell_size
            
#                try:
#                    print(np.array(self.current_pos)==np.array([row,col]).reshape(-1,1))
#                except Exception as e:
#                    print('Initial state')
                
                wind = self.wind[row, col]
                if wind > 0:  # Positive wind
                    pygame.draw.rect(self.screen, (0, 255*wind/(np.max(np.abs(self.wind))), 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                elif wind < 0:  # Negative wind
                    pygame.draw.rect(self.screen, (-255*wind/(np.max(np.abs(self.wind))), 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                else:  # No wind
                    pygame.draw.rect(self.screen, (0, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                
                if np.array_equal(np.array(self.current_pos), np.array([row, col])):  # Agent position
                    pygame.draw.rect(self.screen, (0, 0, 255), (cell_left, cell_top, self.cell_size, self.cell_size))
                if np.array_equal(np.array(self.goal_pos), np.array([row, col])):  # Agent position
                    pygame.draw.rect(self.screen, (0, 255, 255), (cell_left, cell_top, self.cell_size, self.cell_size))
                if np.array_equal(np.array(self.start_pos), np.array([row, col])):  # Agent position
                    pygame.draw.rect(self.screen, (255, 255, 0), (cell_left, cell_top, self.cell_size, self.cell_size))

        pygame.display.update()  # Update the display
        if mode != None:
            pygame.time.wait(1)