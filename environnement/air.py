import numpy as np
import environnement.parametres_air as pa


class Air:
    def __init__(self, vent):
        self.longitude = pa.longitude      #de 0 à 357.5 avec un pas de 2.5
        self.latitude = pa.latitude  #de -90 à 90 avec un pas de 2.5
        self.pressure = pa.pressure
        self.data_vent = vent #vent[time][pressure][longitude][latitude]->[u, v]
        #base de donnée: mesure toute les 6h
        #Pression en hPa

    def get_vent(self, pos: list, time: dict, target) -> list:
        t = int(time['steps']//6)
        low_lon = pa.recherche(self.longitude, pos[1])
        low_lat = pa.recherche(self.latitude, pos[0])
        request_vent = self.data_vent[t : t + 2, :, low_lon : low_lon + 2, low_lat : low_lat + 2]
        request_bounds = {'time': [t, t + 1], 'pressure': self.pressure, 'latitude': [self.latitude[low_lat], self.latitude[low_lat + 1]], 'longitude': [self.longitude[low_lon], self.longitude[low_lon + 1]]}
        vent = pa.interpolation(pos, self.pressure, time['steps'], request_vent, request_bounds)
        angle = np.angle(complex(target[0] - pos[0], target[1] - pos[1]))
        ans = []
        for k in range(len(vent)):
            x = np.linalg.norm(vent[k])
            ans.append([x/(x + 30), np.abs(((np.angle(complex(vent[k][1], vent[k][0])) - angle + np.pi)%(2*np.pi) - np.pi)/np.pi)])
        return np.array(ans)