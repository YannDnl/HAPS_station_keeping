import pygrib
import numpy as np

from environnement.data_access.config import GFS_GRIB_FILE


def get_wind_data():
    grid = {
        'time': [],
        'pressure': [],
        'longitude': [],
        'latitude': []
    }
    grbs = pygrib.open(GFS_GRIB_FILE)
    # build grid
    for grb in grbs:
        lats,lons = grb.latlons()
        # if longitudes and latitudes have not been added to grid
        if len(grid['longitude']) == 0:
            grid['longitude'] = lons
            grid['latitude'] = lats
        # else raise error if longitude or latitude format don't match
        else:
            if len(grid['longitude']) != len(lons) or len(grid['latitude']) != len(lats):
                raise ValueError('Lon/lat format does not match between grib messages')
        time = grb.validDate
        if time not in grid['time']:
            grid['time'].append(time)
        pressure = grb.level
        if pressure not in grid['pressure']:
            grid['pressure'].append(pressure)
    grid['time'].sort()
    grid['pressure'].sort(key=lambda x: -x)
    metadata = {'dataset': 'GFS', 'grid': grid}
    # get data
    shape = (len(grid['time']),len(grid['pressure']),len(grid['longitude']),len(grid['latitude']))
    data = np.zeros(shape)
    for grb in grbs:
        time_index = grid['time'].index(grb.validDate)
        pressure_index = grid['pressure'].index(grb.level)
        data[time_index][pressure_index] = np.transpose(grb.values)
    wind_data = {
        'metadata': metadata,
        'data': data
    }
    return wind_data