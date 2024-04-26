import environnement.ballon as ballon
import environnement.parametres_ballon as pb
import environnement.parametres_air as pa
import random as rd
import numpy as np
import matplotlib.pyplot as plt

class Flotte:
    def __init__(self, n:int, vent:list, time:dict, target:list) -> None:
        self.n = n
        self.list_ballon = []
        self.time = time
        for _ in range(n):
            lat = pa.update_longitude(target[0] + 180000 * rd.gauss(0, 50)/(pb.R * np.pi))
            lon = (target[1] + 180000 * rd.gauss(0, 50)/(pb.R * np.pi))%360
            z = pb.conversion_z_to_p(rd.uniform(15000, 20000))
            self.list_ballon.append(ballon.Ballon(vent, time, [lat, lon], z, target))
        self.target = target
    
    def get_reward(self):
        ans = [self.list_ballon[k].get_reward() for k in range(self.n)]
        return max(ans)

    def next_state(self, actions):
        self.time['steps'] += pb.dt
        if(self.time['steps']%6 == 0):
            pb.update_time(self.time)
        for k in range(self.n):
            self.list_ballon[k].next_state(actions[k])
        return self.get_reward()
    
    def get_inputs(self):
        altitudes = []
        charges = []
        distances = []
        bearings = []
        lights = []
        actions = []
        for k in range(self.n):
            l = self.list_ballon[k].get_inputs()
            l.pop()
            actions.append(l.pop())
            lights.append(l.pop())
            bearings.append(l.pop())
            distances.append(l.pop())
            charges.append(l.pop())
            altitudes.append(l.pop())
        return [altitudes, charges, distances, bearings, lights, actions]

    def plot(self):
        x = [self.list_ballon[i].pos[0] for i in range(self.n)]
        y = [self.list_ballon[i].pos[1] for i in range(self.n)]
        plt.scatter(x, y, color='blue', label='Balloons')

        # Plot the red point at the center
        plt.scatter(self.target[0], self.target[1], color='red', label='Target')

        # Plot the circle around the center
        circle = plt.Circle(self.target, 50000 * 180/(pb.R * np.pi), color='green', fill=False, label='Objective')
        plt.gca().add_patch(circle)

        # Set aspect ratio to equal to get a circular plot
        plt.axis('equal')

        # Add legend
        plt.legend()

        # Show plot
        plt.show()