import numpy as np

##Constante physiques

K = 63162.6             #rapport entre P et mv dans les conditions de pressions et de température étudiés
g = 9.81                #acceleration de la gravite
R = 6371000.            #rayon de la terre en m(6356+6378)/2 * 1000
A = 1013.25             #Pression atmospherique au niveau de la mer en hPa
jours = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

##Constantes du problème

m = 5.                  #masse de la nacelle en kg
V = 100.                #volume du ballon en m3
n = 1.5                 #Capacité thermique massique sur R/M
c = 10000000.           #capacité de la batterie en joule (ici environ 3kWh, 10^7 joules)

##Metadata NOAA database

#Longitude de 0 à 357.5 avec un pas de 2.5
#Latitude de -90 à 90 avec un pas de 2.5
#Pression dans [10., 20., 30., 50., 70., 100., 150., 200., 250., 300., 400., 500., 600., 700., 850., 925., 1000.]
#Temps toutes les 6h datetime.datetime(year, month, day, hour) avec hour dans [0, 6, 12, 18]

##Paramètres du modèle
cd = 0.4                #saut de la fonction reward
T = 100.                #distance caractéristique de la fonction reward
dt = 0.005               #Pas de temps du modèle, doit diviser 6 pour la méthode next_state de la classe ballon
dmv = 0.0005             #Pas de masse volumique
P_out = n * K * V * dmv #Pas de consomation d'énergie
P_in = 0                #Pas de production d'énergie (panneau solaires)

##Données initiales

s0 = 0.5                #charge initiale de la batterie
mv0 = 0.0844            #masse volumique initiale dans le ballon, en kg/m3 telle que le ballon soit stable à l'altitude z0
de0 = 0.                #energie consommé initiale

##Liens entre différentes grandeurs
#z en m, p en hPa, mv en kg.m^-3

def conversion_z_to_p(z):
    return A * np.exp(-z * g/K)

def conversion_z_to_mv(z):
    return conversion_p_to_mv(conversion_z_to_p(z))

def conversion_p_to_z(p):
    return (np.log(A)-np.log(p)) * K/g

def conversion_p_to_mv(p):
    return 100 * p/K

def conversion_mv_to_z(mv):
    return conversion_p_to_z(conversion_mv_to_p(mv))

def conversion_mv_to_p(mv):
    return mv * K/100

def mv_prime(mv):
    return mv + m/V

##Trajectoire

def distance(a, b):
    return R * np.arccos(np.cos((a[0] - b[0]) * np.pi/180) * np.cos((a[1] - b[1]) * np.pi/180))/1000

def f(z, mv):
    return conversion_z_to_mv(z)/mv_prime(mv)

def new_altitude(z: float, dzdt: float, mv:float, dmvdt) -> float:
    t = dt * 3600
    a = dmvdt/mv_prime(mv)
    new_dzdt = dzdt * (1 - t * (a + (1 - a * t) * (a))/2) - t * g * ((1 - f(z, mv)) * (1 - t * (a)) + 1 - f(z + dzdt * t, mv + dmvdt * t))/2
    new_z = z + t * (dzdt + new_dzdt)/2
    #adt = g * (conversion_mv_to_z(mv_prime(mv)) - conversion_mv_to_z(mv_prime(old_mv)))/K
    #f = g * (1 - conversion_z_to_mv(z)/(mv_prime(mv)))
    #new_z = ((4 + adt) * z - 2 * f * ((dt * 3600)**2) - (adt + 2) * old_z)/2
    return conversion_z_to_p(new_z), new_dzdt

##Evolution du temps

def update_time(time):
    time['hour'] += 6
    if(time['hour'] >= 24):
        time['hour'] = time['hour']%24
        time['day'] += 1
        limite = jours[time['month'] - 1]
        if(time['month'] == 2 and time['year']%4 == 0):
            limite += 1
        if(time['day'] > limite):
            time['day'] = 1
            time['month'] += 1
            if(time['month'] > 12):
                time['month'] = 1
                time['year'] += 1