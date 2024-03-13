import environnement.parametres_ballon as pb
import numpy as np

class Ballon:
    def __init__(self) -> None:
        self.m = pb.m     #masse de la nacelle
        self.V = pb.V     #Volume du ballon
        self.c = pb.c     #capacité de la batterie

        self.p = pb.p0    #position
        self.z = pb.z0    #altitude
        self.dz = pb.dz0  #vitesse verticale
        self.mv = pb.mv0  #masse volumique dans le ballon
        self.s = pb.s0    #charge de la batterie (max c)
        self.de = pb.de0  #energie consommé (normalisée) /!\ voir si c'est à t ou t+1

    def get_reward(self, o):
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

#P = mv RT/M avec T = 220K, constante de 10 à 20 km (1)(https://www.lmd.ens.fr/legras/Cours/L3-meteo/stratifN.pdf)
#de plus M est constant jusqu'à 100km à 28.96g/mol (1), donc P = mv * K, avec K = 63.2e3 J/kg