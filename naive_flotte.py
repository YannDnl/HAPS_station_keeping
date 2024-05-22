import environnement.parametres_ballon as pb
import numpy as np
import parametres_entrainement as pe

def main(HAPS, time, target, aero, n, nb_steps, pas=500):
    score_malin = []
    for _ in range(nb_steps):
        time['steps'] += pb.dt
        if(time['steps']%6 == 0):
            pb.update_time(time)
        for k in range(n):
            pos = HAPS.list_ballon[k].pos
            lp = [pos.copy() for _ in range(12000, 20000, pas)]
            for i, alt in enumerate(range(12000, 20000, pas)):
                aero.new_pos(lp[i], pb.conversion_z_to_p(alt), time, pb.dt, target) 

            l_dist = [pb.distance(target, l) for l in lp]
            ind = np.argmin(l_dist)
            
            aero.new_pos(pos, pb.conversion_z_to_p(pas*ind + 12000), time, pb.dt, target)
            if pb.conversion_p_to_z(HAPS.list_ballon[k].z) < pas*ind + 12000:
                HAPS.list_ballon[k].z = pb.conversion_z_to_p(pb.conversion_p_to_z(HAPS.list_ballon[k].z) + 50)
            else:
                HAPS.list_ballon[k].z = pb.conversion_z_to_p(pb.conversion_p_to_z(HAPS.list_ballon[k].z) - 50)
            
            HAPS.trajectory.append([HAPS.list_ballon[k].pos[0], HAPS.list_ballon[k].pos[1], pb.conversion_p_to_z(HAPS.list_ballon[k].z)])
        score_malin.append(HAPS.get_reward())
    HAPS.plot(title = f"naive : distance moyenne {sum(score_malin)/nb_steps}, distance finale {score_malin[-1]}")
