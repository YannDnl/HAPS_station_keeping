import numpy as np
import datetime
import environnement.parametres_ballon as pb
R = pb.R

longitude = np.array([i * 2.5 for i in range(144)])      #de 0 à 357.5 avec un pas de 2.5
latitude = np.array([-90 + i * 2.5 for i in range(73)])  #de -90 à 90 avec un pas de 2.5
pressure = np.array([10., 20., 30., 50., 70., 100., 150., 200., 250., 300., 400., 500., 600., 700., 850., 925., 1000.])

##Récupération du vent

def interpolation(pos, pressure, hour, vent, request_bounds) -> list:#vent liste des liste de vents a interpoler
    ans = []
    test = False
    longueur = 1
    if(len(pressure) == 1):
        test = True
        longueur = 2
    for l in range(len(pressure)):
        sum = np.zeros(2)
        for t in range(len(request_bounds['time'])):
            pt = 1 - np.abs(hour%6 - t * 6)/6.
            for p in range(longueur):
                pp = 1 - test * np.abs(request_bounds['pressure'][p] - pressure[0])/(request_bounds['pressure'][1] - request_bounds['pressure'][0])
                for lon in range(len(request_bounds['longitude'])):
                    plon = 1 - np.abs(request_bounds['longitude'][lon] - pos[1])/2.5
                    for lat in range(len(request_bounds['latitude'])):
                        plat = 1 - np.abs(request_bounds['latitude'][lat] - pos[0])/2.5
                        try :
                            sum += vent[t][l + p][lon][lat] * pt * pp * plon * plat
                        except:
                            print(t, l+p, lon, lat, vent.shape)
                            raise ValueError
        ans.append(sum)
    return ans

def recherche(liste: list, x: float) -> int:
    low = 0
    high = len(liste) - 1
    while(high - low > 1):
        m = liste[(high + low)//2]
        if(m > x):
            high = (high + low)//2
        else:
            low = (high + low)//2
    return low

def date_vent(time):
    start_date = datetime.datetime(year = time['year'], month = time['month'], day = time['day'], hour = time['hour'])
    copy = time.copy()
    pb.update_time(copy)
    end_date = datetime.datetime(year = copy['year'], month = copy['month'], day = copy['day'], hour = copy['hour'])
    return [start_date, end_date]

def update_longitude(teta):
    while(teta > 90 or teta < -90):
        if(teta > 90):
            teta = 180 - teta
        else:
            teta = -180 -teta
    return teta

def get_vent_pos(data_vent, ballon):
        t = int(ballon.time['steps']//6)
        low_lon = recherche(longitude, ballon.pos[1])
        low_lat = recherche(latitude, ballon.pos[0])
        low_p = recherche(pressure, ballon.z)
        request_vent = data_vent[t : t + 2, low_p : low_p + 2, low_lon : low_lon + 2, low_lat : low_lat + 2]
        request_bounds = {'time': [t, t + 1], 'pressure': [pressure[low_p], pressure[low_p + 1]], 'latitude': [latitude[low_lat], latitude[low_lat + 1]], 'longitude': [longitude[low_lon], longitude[low_lon + 1]]}
        vent = interpolation(ballon.pos, [ballon.z], ballon.time['steps'], request_vent, request_bounds)[0]
        vent[0], vent[1] = vent[1], vent[0]
        return vent

def new_pos(vent, pos: list, target: tuple):
    pos[0] += 180 * vent[0] * pb.dt * 3600/(np.pi * R)
    pos[0] = update_longitude(pos[0])
    pos[1] += 180 * vent[1] * pb.dt * 3600/(np.pi * R)
    pos[1] = pos[1]%360
    vector = np.array(target) - np.array(pos)
    return np.array([np.dot(vent, vector), np.cross(vent, vector)])/(np.linalg.norm(vent) * np.linalg.norm(vector))