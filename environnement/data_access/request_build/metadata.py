from environnement.data_access.request_build.grid import create_grid_any, create_grid_date
import numpy as np
from datetime import datetime, timedelta
from environnement.data_access.datasets.datasets_def import datasets_def
from environnement.data_access.coordinates_system import find_longitude_bounds

""" 
request_item = {
        'memory_limit': input("Memory limit in MB: "), 'dataset': input("Dataset: "),
        'bounds': {
            'time': None,
            'pressure': None,
            'longitudes': None,
            'latitudes': None,
        }, 
        'subsampling': {
            'month': None,
            'day': None,
            'time': None,
            'pressure': None,
            'longitude': None,
            'latitude': None
        }
    }
"""
sbs_names = ['month', 'day', 'hour', 'pressure', 'longitude', 'latitude']

def make_metadata_from_dataset(dataset_source,dataset_data):
    """ Makes metadata from a dataset : used for a GET to have metadata that reflect by design what has been get, and for request independant operation that directly read files (e.g storage info)
        Directly takes
        Input:
            dataset_source : str -> name of source
            dataset_data : xarray.Dataset
        Output:
            metadata : dict
    """
    # create grid
    grid = {dim: dataset_data.coords[dim].values for dim in dataset_data.dims}
    # convert timestamp in datetimes
    convert_timestamp_in_datetime = datasets_def[dataset_source]['convert_time']
    grid['time'] = [convert_timestamp_in_datetime(time) for time in grid['time']]
    # compute bounds
    bounds = {dim: [coords[0],coords[-1]] if dim != 'longitude' else find_longitude_bounds(coords) for dim,coords in grid.items()}
    metadata = {'dataset':dataset_source,'grid':grid,'bounds':bounds}
    return metadata

def make_metadata_from_request_item(request_item, compute_bounds=True):
    """ Makes the metadata from one request item
        By default, metadata has three field : 'dataset', 'grid' and 'bounds'
        Finding bounds of dimensions is not possible if empty, that's why an error is raised if a dim is empty
        If flag compute_bounds is False, no error will be raised if corresponding metadata is empty but bounds will not be returned (empty dictionary)"""

    one_day_dimensions = datasets_def[request_item['dataset']]['one_day_dimensions']()
    # if no subsampling in request_item
    if 'subsampling' not in request_item:
        request_item['subsampling'] = {
            "month": 1,
            "day": 1,
            "hour": 1,
            "pressure": 1,
            "longitude": 1,
            "latitude": 1
        }
    else:
        for sbs in sbs_names:
            if sbs not in request_item['subsampling']:
                request_item['subsampling'][sbs] = 1

    """gets the different bounds and subsampling from our parameters"""
    pre_bounds = check_if_list_or_single(request_item['bounds']['pressure'])
    pre_sbs = request_item['subsampling']['pressure']
    # manage the change between (-180,180) and (0,360)
    lon_bounds = check_if_list_or_single(request_item['bounds']['longitude'])
    if lon_bounds[0] < 0 or lon_bounds[-1] < 0:
        if lon_bounds[0] == -180 and lon_bounds[-1] == 180:
            lon_bounds = check_if_list_or_single([0, 360])
        else:
            lon_bounds = check_if_list_or_single([(bound+360)%360 for bound in request_item['bounds']['longitude']])
    else:
        lon_bounds = check_if_list_or_single(request_item['bounds']['longitude'])


    lon_sbs = request_item['subsampling']['longitude']

    lat_bounds = check_if_list_or_single(request_item['bounds']['latitude'])
    lat_sbs = request_item['subsampling']['latitude']

    time_bounds = check_if_list_or_single(request_item['bounds']['time'])
    month_sbs = request_item['subsampling']['month']
    day_sbs = request_item['subsampling']['day']
    hour_sbs = request_item['subsampling']['hour']

    """calculate the grids for every parameters depending on the bounds and subsampling"""
    pre_grid = create_grid_any(pre_bounds, pre_sbs, one_day_dimensions, 'pressure')
    lon_grid = create_grid_any(lon_bounds, lon_sbs, one_day_dimensions, 'longitude')
    lat_grid = create_grid_any(lat_bounds, lat_sbs, one_day_dimensions, 'latitude')
    time_grid = create_grid_date(time_bounds, month_sbs, day_sbs, hour_sbs, one_day_dimensions)

    grid = {
        'pressure': pre_grid,
        'longitude': lon_grid,
        'latitude': lat_grid,
        'time': time_grid
    }

    if compute_bounds:
        for dim in grid:
            if len(grid[dim])==0:
                raise ValueError("There is no value between the two bounds given for " + dim)
        bounds = {
            'pressure': [pre_grid[0], pre_grid[-1]],
            'longitude': [lon_grid[0], lon_grid[-1]],
            'latitude': [lat_grid[0], lat_grid[-1]],
            'time': [time_grid[0], time_grid[-1]]
        }
    else:
        bounds = {}

    metadata = {
        'dataset': request_item['dataset'],
        'grid': grid,
        'bounds': bounds
    }
    # checks if filling parameters entered
    if 'fill_pattern' in request_item:
        fill_metadata(request_item, metadata, one_day_dimensions)
        size = datasets_def[request_item['dataset']]['grid_memory_size'](metadata)
        print(f'After filling the metadata, the new size of the request is {size:0.2f}MB')
    return metadata


def check_if_list_or_single(element):
    """ return the correct np.array by checking whether the input is a list or not
            added another check for time format, it can be a dictionary or a datetime"""
    if isinstance(element, list):
        if isinstance(element[0], dict):
            return np.array([datetime(element[0]['year'],
                                      element[0]['month'],
                                      element[0]['day'],
                                      element[0]['hour']),
                             datetime(element[-1]['year'],
                                      element[-1]['month'],
                                      element[-1]['day'],
                                      element[-1]['hour'])])
        return np.array(element)

    else:
        if isinstance(element, dict):
            return np.array([datetime(element['year'],
                                      element['month'],
                                      element['day'],
                                      element['hour'])])
        return np.array([element])


def fill_metadata(request_item, metadata, one_day_dimensions):
    """ fills the metadata according to the parameters entered
            browse every pattern in request_item, for each of them we fill the metadata by taking care not to add too many values
            in order not to add too much values, we calculate the number of data to add necessary to reach the memory_limit
            then we try to fill those values in metadata and respecting the other parameters such as filling_rule and limit"""
    if not isinstance(request_item['fill_pattern'], list):
        request_item['fill_pattern'] = [request_item['fill_pattern']]
    for pattern in request_item['fill_pattern']:
        dimension_fill = pattern['dimension']
        size_request = datasets_def[request_item['dataset']]['grid_memory_size'](metadata)
        if size_request < request_item['memory_limit']:
            size_one_data = 1
            for name in metadata['grid']:
                if name != dimension_fill:
                    size_one_data *= len(metadata['grid'][name])
            size_one_data = size_one_data * 2 * 2 / (1024 ** 2)
            number_data_to_add = int((request_item['memory_limit']-size_request)/size_one_data)
            if dimension_fill == 'time':
                metadata['grid'][dimension_fill] = fill_with_rule_time(pattern, request_item,
                                                                       one_day_dimensions, metadata, number_data_to_add)
            else:
                metadata['grid'][dimension_fill] = fill_with_rule(pattern, request_item,
                                                                  one_day_dimensions, metadata, number_data_to_add)
        metadata['bounds'][dimension_fill] = [metadata['grid'][dimension_fill][0], metadata['grid'][dimension_fill][-1]]
    return metadata


def fill_with_rule(pattern, request_item, one_day_dimensions, metadata, number_data_to_add):
    """ revoir le fonctionnement pour la longitude vérifier si on ne fait pas plus qu'un tour pour tout prendre"""
    filling_rule = pattern['filling_rule']
    dimension_fill = pattern['dimension']
    subsampling = request_item['subsampling'][dimension_fill]
    if filling_rule not in ['forward', 'backward', 'symmetric']:
        msg = f"Filling rule must be one of these ['forward', 'backward', 'symmetric']"
        raise ValueError(msg)

    one_day_dimensions_sbs = one_day_dimensions[dimension_fill][::subsampling]
    if filling_rule == 'forward':
        if number_data_to_add > len(one_day_dimensions_sbs) and dimension_fill=='longitude':
            grid = one_day_dimensions_sbs
        else:
            index_sup = np.where(metadata['grid'][dimension_fill][-1] == one_day_dimensions_sbs)[0][0]
            grid = np.concatenate((metadata['grid'][dimension_fill],one_day_dimensions_sbs[index_sup+1:index_sup+number_data_to_add]))
        if 'limit' in pattern:
            grid = grid[grid<=pattern['limit']]
    elif filling_rule == 'backward':
        if number_data_to_add > len(one_day_dimensions_sbs) and dimension_fill=='longitude':
            grid = one_day_dimensions_sbs
        else:
            index_inf = np.where(metadata['grid'][dimension_fill][0] == one_day_dimensions_sbs)[0][0]
            grid = np.concatenate((one_day_dimensions_sbs[index_inf-number_data_to_add:index_inf],metadata['grid'][dimension_fill]))
        if 'limit' in pattern:
            grid = grid[grid>=pattern['limit']]
    elif filling_rule == 'symmetric':
        if number_data_to_add > len(one_day_dimensions_sbs) and dimension_fill=='longitude':
            grid = one_day_dimensions_sbs
        else:
            index_inf = np.where(metadata['grid'][dimension_fill][0] == one_day_dimensions_sbs)[0][0]-int(number_data_to_add/2)
            index_sup = np.where(metadata['grid'][dimension_fill][-1] == one_day_dimensions_sbs)[0][0]+int(number_data_to_add/2)
            if index_inf>index_sup:
                grid = np.concatenate((one_day_dimensions_sbs[index_inf:], one_day_dimensions_sbs[:index_sup]))
            else:
                grid = one_day_dimensions_sbs[index_inf:index_sup+1]
            if 'limit' in pattern:
                if not isinstance(pattern['limit'], list):
                    raise ValueError("For a symmetric filling_rule, 'limit' must be a list of two values")
                grid = grid[(grid<=pattern['limit'][-1]) & (grid>=pattern['limit'][0])]
    grid.sort()
    return grid


def fill_with_rule_time(pattern, request_item, one_day_dimensions, metadata, number_data_to_add):
    filling_rule = pattern['filling_rule']
    month_sbs = request_item['subsampling']['month']
    day_sbs = request_item['subsampling']['day']
    hour_sbs = request_item['subsampling']['hour']
    number_hours_to_add = number_data_to_add*hour_sbs*day_sbs*month_sbs - 0.05*number_data_to_add*hour_sbs*day_sbs*month_sbs
    if 'limit' in pattern:
        if number_hours_to_add > pattern['limit'].total_seconds()/3600:
            number_hours_to_add = pattern['limit'].total_seconds()/3600
    if filling_rule not in ['forward', 'backward', 'symmetric']:
        msg = f"Filling rule must be one of these ['forward', 'backward', 'symmetric']"
        raise ValueError(msg)
    if filling_rule == 'forward':
        grid_date = create_grid_date([metadata['bounds']['time'][0], metadata['bounds']['time'][-1] + timedelta(hours=number_hours_to_add)],
                                     month_sbs,
                                     day_sbs,
                                     hour_sbs,
                                     one_day_dimensions)
    elif filling_rule == 'backward':
        grid_date = create_grid_date([metadata['bounds']['time'][0] - timedelta(hours=number_hours_to_add), metadata['bounds']['time'][-1]],
                                     month_sbs,
                                     day_sbs,
                                     hour_sbs,
                                     one_day_dimensions)
    elif filling_rule == 'symmetric':
        grid_date = create_grid_date([metadata['bounds']['time'][int(len(metadata['bounds']['time'])/2)] - timedelta(hours=int(number_hours_to_add/2)),
                                      metadata['bounds']['time'][int(len(metadata['bounds']['time'])/2)] + timedelta(hours=int(number_hours_to_add/2))],
                                     month_sbs,
                                     day_sbs,
                                     hour_sbs,
                                     one_day_dimensions)
    return grid_date