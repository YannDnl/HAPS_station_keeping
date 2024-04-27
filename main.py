import environnement.flotte as flotte
import agent.agent as agent
import parametres_entrainement as pe

import environnement.data as data

time = {'year': 2020, 'month': 1, 'day': 1, 'hour': 0, 'steps': 0}

target, start_time = pe.position_initiale(time)
donnee_vent = data.get_data(start_time)

n = 10
size = 1
t = 1000
pilote_rd = agent.RandomAgent(n)
pilote_p = agent.PassiveAgent(n)

l = []

for _ in range(size):
    target, start_time = pe.position_initiale(time)
    donnee_vent = data.get_data(start_time)
    HAPS = flotte.Flotte(n, donnee_vent, start_time, target, show = True)
    #HAPS.plot()
    ans = 0
    for _ in range(t):
        ans += HAPS.next_state(pilote_p.get_action(HAPS.get_inputs()))//1
    l.append(ans/t)
    #HAPS.plot()
    HAPS.plot()
print(l)
print(sum(l)/size)