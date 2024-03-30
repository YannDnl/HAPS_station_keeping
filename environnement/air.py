import numpy as np
import parametres as pb
import datetime

class Air:
    def __init__(self, start_date = pb.start_date0):
        self.vent = []
        self.longitude = np.array([i * 2.5 for i in range(144)])      #de 0 à 357.5 avec un pas de 2.5
        self.latitude = np.array([-90 + i * 2.5 for i in range(73)])  #de -90 à 90 avec un pas de 2.5
        self.pressure = np.array([10., 20., 30., 50., 70., 100., 150., 200., 250., 300., 400., 500., 600., 700., 850., 925., 1000.])
        self.altitude = pb.conversion_p_to_z(self.pressure)
        self.time = datetime.datetime(year = 2020, month = 1, day = 1, hour = 0)

    def get_vent(self, pos: tuple) -> list:
        return []
    
    def new_pos(self, pos: tuple) -> tuple:
        return (0, 0)
    