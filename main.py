import random as rd
import environnement.ballon as ballon
import agent.agent as agent

objectif = (rd.uniform(-90, 90), rd.uniform(0, 360))

haps = ballon.Ballon(pos = (rd.uniform(-90, 90), rd.uniform(0, 360)))
pilote = agent.Agent()

#for _ in range (100):
#    action = pilote.get_action(haps.get_inputs)
#    haps.next_state(action)