import environnement.parametres_ballon as pb
import environnement.air as air
import numpy as np

class Ballon:
    def __init__(self, vent:list, time:dict, pos:list, z:float, target:tuple, seul = False) -> None:
        self.pos = pos                      #position: latitude puis longitude
        self.z = z                          #altitude par la pression
        self.dzdt = 0                      #altitude au pas de temps précédent
        self.mv = pb.conversion_p_to_mv(z) - pb.m/pb.V  #masse volumique dans le ballon
        self.dmvdt = 0               #masse volumique dans le ballon au pas de temps précédent
        self.s = pb.s0                      #charge de la batterie (max c)
        self.de = pb.de0                    #energie consommé (normalisée) /!\ voir si c'est à t ou t+1
        self.time = time                    #dictionnaire des coordonées temporelles
        self.soleil = False                 #s'il y a du soleil, ie si la batterie se recharge
        self.update_soleil()
        self.target = target
        self.bearing = [0, 0]
        self.last_action = 0
        self.seul = seul

        self.air = air.Air(vent)

    def copy(self):
        return Ballon(self.air.data_vent, self.time, self.pos.copy(), self.z, self.target, self.seul)
    
    def get_reward(self, dist_inv = True) -> float:

        delta = pb.distance(self.target, self.pos)

        if dist_inv:
            return -delta
        
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
        self.last_action = action
        if(self.seul):
            self.time['steps'] += pb.dt
            if(self.time['steps']%6 == 0):
                pb.update_time(self.time)
        if(self.soleil and self.s < pb.c):
            self.s = min(pb.c, self.s + pb.P_in)
        self.update_soleil()
        self.bearing = self.air.new_pos(self.pos, self.z, self.time, pb.dt, self.target)
        if(action * (pb.conversion_mv_to_p(pb.mv_prime(self.mv)) - self.z) >= 0):
            self.de = pb.P_out
        else:
            self.de = 0
        if(self.de <= self.s):
            self.s -= self.de
            new_mv = max(0, self.mv + (-1) * action * pb.dmv)
        else:
            self.de = 0
            new_mv = self.mv
        self.z, self.dzdt = pb.new_altitude(pb.conversion_p_to_z(self.z), self.dzdt, new_mv, self.dmvdt)
        self.dmvdt = (new_mv - self.mv)/(pb.dt * 3600)
        self.mv = new_mv
        return self.get_reward()

    def update_soleil(self) -> None:
        if((self.time['hour'] - 6)%24 < 12):
            self.soleil = True
        else:
            self.soleil = False

    def get_inputs(self):#inclure air et reward, tout normaliser
        ans = []
        ans.append((self.z - pb.conversion_z_to_p(20000))/(pb.conversion_z_to_p(15000) - pb.conversion_z_to_p(20000)))
        ans.append(self.s/pb.c)
        d = pb.distance(self.target, self.pos)
        ans.append(d/(d + 250000))
        ans.append(self.bearing)
        ans.append(self.soleil)
        ans.append(self.last_action)
        ans.append(self.air.get_vent(self.pos, self.time, self.target))
        return ans


#P = mv RT/M avec T = 220K, constante de 10 à 20 km (1)(https://www.lmd.ens.fr/legras/Cours/L3-meteo/stratifN.pdf)
#de plus M est constant jusqu'à 100km à 28.96g/mol (1), donc P = mv * K, avec K = 63.2e3 J/kg