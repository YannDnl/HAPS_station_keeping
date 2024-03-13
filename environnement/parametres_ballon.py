##Constante physiques

K = 63162.6 #rapport entre P et mv dans les conditions de pressions et de température étudiés
g = 9.81    #acceleration de la gravite
R = 6371    #rayon de la terre en km

##Paramètres du modèle
cd = 0.4    #saut de la fonction reward
T = 100     #distance caractéristique de la fonction reward

##Constantes du problème

m = 100             #masse en kg
V = 100             #volume en m3
c = 10**7           #capacité de la batterie en joule (ici environ 3kWh)

##Données initiales

s0 = 0.5      #charge initiale de la batterie
p0 = (45, 0)  #latitude (S-N) dans (-90, 90) puis longitude (E-W) dans (-180, 180) initiales
z0 = 10       #altitude mesuré par la pression en kPa allant 5 kPa (20km) to 14 kPa (15km)
dz0 = 0       #vitesse verticale initiale
mv0 = 0.158   #masse volumique initiale dans le ballon, en kg/m3 telle que le ballon soit stable à l'altitude z0
de0 = 0       #energie consommé initiale