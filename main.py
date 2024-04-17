import environnement.ballon as ballon
import agent.agent as agent
import parametres_entrainement as pe

import environnement.data as data

time = {'year': 2020, 'month': 1, 'day': 1, 'hour': 0, 'steps': 0}

donnee_vent = data.get_data(time)

pos, z, index, start_time = pe.position_initiale(time)

HAPS = ballon.Ballon(donnee_vent[index:], start_time, pos, z, pos)
#pilote = agent.Agent()

#for _ in range (100):
#    action = pilote.get_action(haps.get_inputs)
#    haps.next_state(action)