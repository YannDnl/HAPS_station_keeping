import random as rd
import environnement.ballon as ballon
import agent.agent as agent
import parametres_entrainement as pe
import numpy as np
import environnement.parametres_ballon as pb

import environnement.data as data

jours = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

time = {'year': 2020, 'month': 1, 'day': 1, 'hour': 0, 'steps': 0}

donnee_vent = data.get_data(time)

#choisir un index uniformément dans (0, (365 - (duree d'une session d'entrainement en jours)) * 4 - 1) pour le start

def position_initiale(time): ##renvoie une position et un instant initiale aléatoire
    index = rd.randint(0, (365 - int(pe.duration)) * 4)
    days = (index + time['hour']//6)//4
    i = -1
    sum = jours[time['month'] + i] - time['day'] + 1
    while(sum < days):
        i += 1
        sum += jours[(time['month'] + i)%12]
    sum -= jours[(time['month'] + i)%12]
    sum = np.max([0, sum])
    start_time = time
    start_time['day'] += days - sum
    start_time['month'] = (time['month'] + i + 1)%12
    start_time['year'] += (time['month'] + i)//12
    start_time['hour'] = (index * 6 + time['hour'])%24
    pos = []
    pos.append(180 * np.arcsin(2 * rd.uniform(0, 1) - 1)/np.pi)
    pos.append(rd.uniform(0, 360))
    pressure = pb.conversion_z_to_p(rd.uniform(15000, 20000))
    return pos, pressure, index, start_time


#HAPS = Ballon(donnee_vent[index:], start_time, pos, pressure, pos)
#pilote = agent.Agent()

#for _ in range (100):
#    action = pilote.get_action(haps.get_inputs)
#    haps.next_state(action)