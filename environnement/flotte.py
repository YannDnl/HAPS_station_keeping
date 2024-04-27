import environnement.ballon as ballon
import environnement.parametres_ballon as pb
import environnement.parametres_air as pa
import random as rd
import numpy as np
import matplotlib.pyplot as plt

class Flotte:
    def __init__(self, n:int, vent:list, time:dict, target:list, show = False) -> None:
        self.n = n
        self.list_ballon = []
        self.trajectory = []
        self.show = show
        self.time = time
        for _ in range(n):
            lat = pa.update_longitude(target[0] + 180000 * rd.gauss(0, 50)/(pb.R * np.pi))
            lon = (target[1] + 180000 * rd.gauss(0, 50)/(np.cos(lat * np.pi/180) * pb.R * np.pi))%360
            z = pb.conversion_z_to_p(rd.uniform(15000, 20000))
            self.list_ballon.append(ballon.Ballon(vent, time, [lat, lon], z, target))
            if(self.show):
                self.trajectory.append([lat, lon, pb.conversion_p_to_z(z)])
        self.target = target
    
    def get_reward(self):
        ans = [self.list_ballon[k].get_reward() for k in range(self.n)]
        return np.max(ans)

    def next_state(self, actions):
        self.time['steps'] += pb.dt
        if(self.time['steps']%6 == 0):
            pb.update_time(self.time)
        for k in range(self.n):
            self.list_ballon[k].next_state(actions[k])
            if(self.show):
                self.trajectory.append([self.list_ballon[k].pos[0], self.list_ballon[k].pos[1], pb.conversion_p_to_z(self.list_ballon[k].z)])
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
        if(self.show):
            fig, (ax1, ax2) = plt.subplots(1, 2)
            for k in range(self.n):
                l = [self.trajectory[k + i * self.n] for i in range(len(self.trajectory)//self.n)]
                lat = [l[i][1] for i in range(len(l))]
                lon = [l[i][0] for i in range(len(l))]
                z = [l[i][2] for i in range(len(l))]
                ax1.scatter(lat[0], lon[0], color = 'blue', label='Position initiale')
                ax1.scatter(lat[-1], lon[-1], color = 'orange', label='Position finale')
                ax1.plot(lat, lon)
                ax2.plot(z)
            ax1.set_title('Trajectoire')
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')
            ax1.scatter(self.target[1], self.target[0], color='red', label='Target')
            circle = plt.Circle([self.target[1], self.target[0]], 50000 * 180/(pb.R * np.pi), color='green', fill=False, label='Objective')
            ax1.add_patch(circle)
            ax1.set_aspect('equal')

            ax2.set_title('Altitude')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Altitude')
            plt.tight_layout()
        else:
            lat = [self.list_ballon[i].pos[0] for i in range(self.n)]
            lon = [self.list_ballon[i].pos[1] for i in range(self.n)]
            plt.scatter(lon, lat, color='blue', label='Balloons')

            # Plot the red point at the center
            plt.scatter(self.target[1], self.target[0], color='red', label='Target')

            # Plot the circle around the center
            circle = plt.Circle([self.target[1], self.target[0]], 50000 * 180/(pb.R * np.pi), color='green', fill=False, label='Objective')
            plt.gca().add_patch(circle)

            # Set aspect ratio to equal to get a circular plot
            plt.axis('equal')

            plt.xlabel('Longitude')
            plt.ylabel('Latitude')

            # Add legend
            plt.legend()

        # Show plot
        plt.show()