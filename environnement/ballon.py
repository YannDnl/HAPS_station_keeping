import environnement.parametres as pb
import air
import numpy as np

class Ballon:
    def __init__(self) -> None:
        float: self.m = pb.m               #masse de la nacelle
        float: self.V = pb.V               #Volume du ballon
        float: self.c = pb.c               #capacité de la batterie

        tuple: self.p = pb.p0              #position
        float: self.z = pb.z0              #altitude
        float: self.old_z = self.z         #altitude au pas de temps précédent
        float: self.mv = pb.mv0            #masse volumique dans le ballon
        float: self.old_mv = self.mv       #masse volumique dans le ballon au pas de temps précédent
        float: self.s = pb.s0              #charge de la batterie (max c)
        float: self.de = pb.de0            #energie consommé (normalisée) /!\ voir si c'est à t ou t+1

        self.air = air.Air()

    def get_reward(self, o:tuple) -> float:
        delta = pb.R * np.arccos(np.cos((o[0] - self.p[0]) * np.pi/180) * np.cos((o[1] - self.p[1]) * np.pi/180))
        if delta < 50:
            f = 1.0
        else:
            f = pb.cd * np.exp((50 - delta)/pb.T)
        if self.de == 0:
            r = 1
        else:
            r = 0.95 - 0.3 * self.de
        return f * r
    
    def next_state(self, action:int) -> None:
        self.p = self.air.new_pos(self.p)
        if(action * (pb.conversion_mv_to_z(pb.mv_prime(self.mv)) - self.z) >= 0):
            self.de = pb.de
        else:
            self.de = 0
        if(self.de <= self.s):
            self.s -= self.de
            new_mv = self.mv + (-1) * action * pb.dmv
        else:
            self.de = 0
            new_mv = self.mv
        self.z, self.old_z = pb.new_altitude(self.z, self.old_z, new_mv, self.old_mv), self.z
        self.old_mv = self.mv
        self.mv = new_mv


#P = mv RT/M avec T = 220K, constante de 10 à 20 km (1)(https://www.lmd.ens.fr/legras/Cours/L3-meteo/stratifN.pdf)
#de plus M est constant jusqu'à 100km à 28.96g/mol (1), donc P = mv * K, avec K = 63.2e3 J/kg