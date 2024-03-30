import numpy as np

##Constante physiques

K = 63162.6             #rapport entre P et mv dans les conditions de pressions et de température étudiés
g = 9.81                #acceleration de la gravite
R = 6371.0              #rayon de la terre en km
A = 101325.0            #Pression atmospherique au niveau de la mer

##Constantes du problème

m = 100.0               #masse de la nacelle en kg
V = 100.0               #volume du ballon en m3
n = 3/2                 #Capacité thermique massique sur R/M
c = 10000000.0          #capacité de la batterie en joule (ici environ 3kWh, 10^7 joules)

##Paramètres du modèle
cd = 0.4                #saut de la fonction reward
T = 100.0               #distance caractéristique de la fonction reward
dt = 1.0                #Pas de temps du modèle
dmv = 0.01              #Pas de masse volumique
de = n * K * V * dmv    #Pas d'énergie

##Données initiales

s0 = 0.5                #charge initiale de la batterie
p0 = (45.0, 0.0)        #latitude (S-N) dans (-90, 90) puis longitude (E-W) dans (-180, 180) initiales
z0 = 10.0               #altitude mesuré par la pression en kPa allant 5 kPa (20km) to 14 kPa (15km)
mv0 = 0.158             #masse volumique initiale dans le ballon, en kg/m3 telle que le ballon soit stable à l'altitude z0
de0 = 0.0               #energie consommé initiale

##Liens entre différentes grandeurs

def conversion_z_to_p(z):
    return A * np.exp(-z * g/K)

def conversion_z_to_mv(z):
    return conversion_p_to_mv(conversion_z_to_p(z))

def conversion_p_to_z(p):
    return np.log(A/p) * K/g

def conversion_p_to_mv(p):
    return p/K

def conversion_mv_to_z(mv):
    return conversion_p_to_z(conversion_mv_to_p(mv))

def conversion_mv_to_p(mv):
    return mv * K

def mv_prime(mv):
    return mv + m/V

##Trajectoire

def new_altitude(z: float, old_z: float, mv:float, old_mv) -> float:
    adt = g * (conversion_mv_to_z(mv_prime(mv)) - conversion_mv_to_z(mv_prime(old_mv)))/K
    f = g * (1 - np.exp(-g * (z - conversion_mv_to_z(mv_prime(mv)))/K))
    ans = (4 * z - 2 * f * (dt**2) + - (adt + 2) * old_z)/(2 - adt)
    return ans