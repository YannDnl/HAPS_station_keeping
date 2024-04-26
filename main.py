import environnement.flotte as flotte
import agent.agent as agent
import parametres_entrainement as pe

import environnement.data as data

time = {'year': 2020, 'month': 1, 'day': 1, 'hour': 0, 'steps': 0}

target, start_time = pe.position_initiale(time)

donnee_vent = data.get_data(start_time)

n = 10

HAPS = flotte.Flotte(n, donnee_vent, start_time, target)
pilote = agent.Agent(n)
HAPS.plot()
for _ in range(3):
    HAPS.next_state(pilote.get_action(HAPS.get_inputs()))

HAPS.plot()
#pilote = agent.Agent()

#for _ in range (100):
#    action = pilote.get_action(haps.get_inputs)
#    haps.next_state(action)