import numpy as np


def find_longitude_bounds(longitude,min_lon_value=0):
    longitude = min_lon_value+(np.array(longitude)-min_lon_value)%360
    longitude = np.sort(longitude)
    max_hole = max((np.roll(longitude,-1)-longitude)%360)
    if (longitude[0]-longitude[-1])%360 == max_hole:
        i_upper_bound, i_lower_bound = -1,0
    else:
        i_upper_bound = np.argmax((np.roll(longitude,-1)-longitude)%360)
        i_lower_bound = (i_upper_bound+1)%len(longitude)
    bounds = (float(longitude[i_lower_bound]),float(longitude[i_upper_bound]))
    return bounds