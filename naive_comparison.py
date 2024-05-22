import environnement.flotte as flotte

import environnement.data as data
import environnement.air as air
import parametres_entrainement as pe
import environnement.parametres_ballon as pb
import passive
import naive
import naive_yd
import naive_ak
import naive_em
import numpy as np
import copy

n = 1

start_time = {'year': 2020, 'month': 1, 'day': 1, 'hour': 0, 'steps': 0}
target = [1600, 0]
while np.abs(target[0]) > 10:
    target, time = pe.position_initiale(start_time)
print('conditions :', target, time)

vent = data.get_data(time)
aero = air.Air(vent)

pas = 100
pe.nb_steps = 25000
pb.dt = 0.005

HAPS = flotte.Flotte(n, vent, time, target, show = True)
passive.main(HAPS.copy(), n, pe.nb_steps)
pe.nb_steps = 2500
pb.dt = 0.05
naive.main(HAPS.copy(), copy.deepcopy(time), target, aero, n, pe.nb_steps, pas=pas)
naive_yd.main(HAPS.copy(), copy.deepcopy(time), target, aero, n, pe.nb_steps, pas=pas)
naive_ak.main(HAPS.copy(), copy.deepcopy(time), target, aero, n, pe.nb_steps, pas=pas)
#naive_em.main(HAPS.copy(), copy.deepcopy(time), target, aero, n, pe.nb_steps, pas=pas)

