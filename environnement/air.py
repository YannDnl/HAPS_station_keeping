import numpy as np
import environnement.parametres_air as pa
import environnement.data as data

class Air:
    def __init__(self):
        self.request_item = {
            'dataset': "NOAA",
            'memory_limit': 200*1000,
            'bounds': {
                'latitude': [],
                'longitude': [],
                'pressure':[],
                'time': []
            },
            'subsampling':{
                'longitude':1,
                'latitude':1,
                'pressure':1,
                'hour':1,
                'month':1,
                'day':1
            }
        }
        self.longitude = np.array([i * 2.5 for i in range(144)])      #de 0 à 357.5 avec un pas de 2.5
        self.latitude = np.array([-90 + i * 2.5 for i in range(73)])  #de -90 à 90 avec un pas de 2.5
        self.pressure = np.array([10., 20., 30., 50., 70., 100., 150., 200., 250., 300., 400., 500., 600., 700., 850., 925., 1000.])
        #Pression en hPa

    def get_vent(self, pos: tuple, time: float) -> list:
        self.request_item['bounds']['latitude'] = pa.recherche(self.latitude, pos[0])
        self.request_item['bounds']['longitude'] = pa.recherche(self.longitude, pos[1])
        self.request_item['bounds']['pressure'] = self.pressure
        self.request_item['bounds']['time'] = pa.date_vent(time)
        vent = pa.interpolation(pos, self.pressure, time['hour'], data.request_vent(self.request_item), self.request_item['bounds'], time)
        return vent
    
    def new_pos(self, pos: tuple, pressure: float, time: float, dt: float):
        self.request_item['bounds']['latitude'] = pa.recherche(self.latitude, pos[0])
        self.request_item['bounds']['longitude'] = pa.recherche(self.longitude, pos[1])
        self.request_item['bounds']['pressure'] = pa.recherche(self.pressure, pressure)
        self.request_item['bounds']['time'] = pa.date_vent(time)
        vent = pa.interpolation(pos, [pressure], time['hour'], data.request_vent(self.request_item), self.request_item['bounds'], time)
        pos[0] += 180 * vent[1] * dt/(np.pi * pa.R)
        pos[1] += 180 * vent[0] * dt/(np.pi * pa.R)