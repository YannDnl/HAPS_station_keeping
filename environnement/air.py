import numpy as np
import environnement.parametres_air as pa


class Air:
    def __init__(self, vent):
        self.longitude = np.array([i * 2.5 for i in range(144)])      #de 0 à 357.5 avec un pas de 2.5
        self.latitude = np.array([-90 + i * 2.5 for i in range(73)])  #de -90 à 90 avec un pas de 2.5
        self.pressure = np.array([10., 20., 30., 50., 70., 100., 150., 200., 250., 300., 400., 500., 600., 700., 850., 925., 1000.])
        self.data_vent = vent #vent[time][pressure][longitude][latitude]
        #Pression en hPa

    def get_vent(self, pos: tuple, time: dict, target) -> list:
        t = time['steps']//6
        low_lon = pa.recherche(self.longitude, pos[1])
        low_lat = pa.recherche(self.latitude, pos[0])
        request_vent = vent[t : t + 2][:][low_lon : low_lon + 2][low_lat : low_lat + 2]
        request_bounds = {'time': [t, t + 1], 'pressure': self.pressure, 'latitude': [self.latitude[low_lat], self.latitude[low_lat + 1]], 'longitude': [self.longitude[low_lon], self.longitude[low_lon + 1]]}
        vent = pa.interpolation(pos, self.pressure, time['steps'], request_vent, request_bounds)
        angle = np.angle(complex(target[0] - pos[0], target[1] - pos[1]))
        ans = []
        for k in range(len(vent)):
            x = np.linalg.norm(vent[k])
            ans.append([x/(x + 30), ((np.angle(complex(vent[k][1], vent[k][0])) - angle)/np.pi)%1])
        return ans
    
    def new_pos(self, pos: tuple, pressure: float, time: dict, dt: float, target: tuple):
        t = time['steps']//6
        low_lon = pa.recherche(self.longitude, pos[1])
        low_lat = pa.recherche(self.latitude, pos[0])
        low_p = pa.recherche(self.pressure, pressure)
        request_vent = vent[t : t + 2][low_p : low_p + 2][low_lon : low_lon + 2][low_lat : low_lat + 2]
        request_bounds = {'time': [t, t + 1], 'pressure': [self.pressure[low_p], self.pressure[low_p + 1]], 'latitude': [self.latitude[low_lat], self.latitude[low_lat + 1]], 'longitude': [self.longitude[low_lon], self.longitude[low_lon + 1]]}
        vent = pa.interpolation(pos, [pressure], time['steps'], request_vent, request_bounds)
        vent[0], vent[1] = vent[1], vent[0]
        pos[0] += 180 * vent[0] * dt/(np.pi * pa.R)
        pos[1] += 180 * vent[1] * dt/(np.pi * pa.R)
        vector = target - pos
        return np.array([np.dot(vent, vector), np.cross(vent, vector)])/(np.linalg.norm(vent) * np.linalg.norm(vector))