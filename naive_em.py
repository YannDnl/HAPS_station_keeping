import environnement.parametres_ballon as pb
import numpy as np
import parametres_entrainement as pe
import environnement.parametres_air as pa

p_angle = 2
def main(HAPS, time, target, aero, n, nb_steps, p_vitesse = 0.3, dist_passience = 50, pas = 500):


    altitude = pb.conversion_p_to_z(aero.pressure)[3:7]
    pressure = pb.conversion_z_to_p(np.array([alt for alt in range(12000, 20000, pas)]))
    ind_alt = [0 if alt<altitude[1] else (1 if alt<altitude[2] else 2) for alt in range(12000, 20000, pas)]
    score_malin = []
    for _ in range(nb_steps):
        time['steps'] += pb.dt
        if(time['steps']%6 == 0):
            pb.update_time(time)
        for k in range(n):
            pos = HAPS.list_ballon[k].pos
            colonne_vraie = aero.data_vent
            t = int(time['steps']/6)
            lat = pa.recherche(aero.latitude, pos[0])
            lon = pa.recherche(aero.longitude, pos[1])
            angle = np.angle(complex(target[0] - pos[0], target[1] - pos[1]))
            ans = []
            colonne_interp = np.array([pa.interpolation(pos, [pressure[i]], time['steps'], colonne_vraie[t:t+2, indice: indice + 2, lon:lon + 2, lat:lat + 2],  {'time': [t, t + 1], 'pressure': [aero.pressure[indice], aero.pressure[indice + 1]], 'latitude': [aero.latitude[lat], aero.latitude[lat + 1]], 'longitude': [aero.longitude[lon], aero.longitude[lon + 1]]})[0] for i, indice in enumerate(ind_alt)])
            for k2fat in range(len(colonne_interp)):
                x = np.linalg.norm(colonne_interp[k2fat])
                ans.append([x, np.abs(((np.angle(complex(colonne_interp[k2fat][1], colonne_interp[k2fat][0])) - angle + np.pi)%(2*np.pi) - np.pi)/np.pi)])
            colonne_interp = np.array(ans)

            if pb.distance(pos, target) < dist_passience:
                i = np.argmax(np.cos(colonne_interp[:,1]*np.pi)*(colonne_interp[:,0]**p_vitesse))
            else:
                i = np.argmax(np.cos(colonne_interp[:,1]*np.pi)*colonne_interp[:,0])
            p_objectif = pressure[i]
            p = HAPS.list_ballon[k].z

    # score_malin = []

    # for _ in range(pe.nb_steps):
    #     time['steps'] += pb.dt
    #     if(time['steps']%6 == 0):
    #         pb.update_time(time)
    #     for k in range(n):
    #         pos = HAPS.list_ballon[k].pos
    #         colonne = aero.get_vent(pos, time, target)[3:7,:]

    #         if pb.distance(pos, target) > dist_passience:
    #             i = np.argmax(((np.cos(colonne[:,1]*np.pi)/2+.5)**p_angle - 0.5**p_angle )*(colonne[:,0]**p_distance))
    #         else:
    #             i = np.argmax(np.cos(colonne[:,1]*np.pi)*colonne[:,0])
    #         p_objectif = aero.pressure[i+3]
    #         p = HAPS.list_ballon[k].z            

            if(p < p_objectif):
                #descendre
                changement = -50
            else:
                changement = 50
            HAPS.list_ballon[k].z = pb.conversion_z_to_p(pb.conversion_p_to_z(p) + changement)

            aero.new_pos(pos, HAPS.list_ballon[k].z, time, pb.dt, target)
            HAPS.trajectory.append([HAPS.list_ballon[k].pos[0], HAPS.list_ballon[k].pos[1], pb.conversion_p_to_z(HAPS.list_ballon[k].z)])
        score_malin.append(HAPS.get_reward())

    HAPS.plot(title = f"naive_em : distance moyenne {sum(score_malin)/nb_steps}, distance finale {score_malin[-1]}")