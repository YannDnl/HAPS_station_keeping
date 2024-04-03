import os
import re
import calendar
import json
import xarray as xr
import numpy as np

from environnement.data_access.datasets.datasets_def import datasets_def
from environnement.data_access.config import STORAGE_DISPLAY_INFO_FILE, LOCAL_STORAGE_DIR
from environnement.data_access.coordinates_system import find_longitude_bounds
from environnement.data_access.request_build.metadata import make_metadata_from_dataset


bound_europe = {'longitude': [350, 40],
                'latitude': [35, 70]}

bound_france = {'longitude': [355, 8.5],
                'latitude': [42, 51]}


def atoi(text):
    """ converts text in int if the text is a number"""
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """ creates natural key for sorting, will be used in .sort()
            if text = 'ERA5.2000.1.2, output ['ERA5',2000,1,2]"""
    return [atoi(c) for c in re.split(r'[.]', text)]



def print_file_info(file_name, storage_info, indent):
    """ prints file information using the storage_info information"""
    info = storage_info[file_name.split('.')[0]][file_name]
    dataset = file_name.split('.')[0]
    convert_timestamp_in_datetime = datasets_def[dataset]['convert_time']
    print('{}{}:    size: {:0.2f} MB, total storage: {:0.2f}%, coverage: world {:0.2f}%, europe {:0.2f}%, france {:0.2f}%,'
          ' bounding_box : time {} {:0.2f}%, pressure {} {:0.2f}%, longitude {} {:0.2f}%, latitude {} {:0.2f}%'.format(
        indent, file_name,
        info['size'],
        info['percentage'],
        info['world'],
        info['europe'],
        info['france'],
        [convert_timestamp_in_datetime(timestamp).strftime('%Y-%m-%dT%H:%M') for timestamp in info['bounding_box_bounds']['time']],
        info['bounding_box_percentage']['time'],
        info['bounding_box_bounds']['pressure'],
        info['bounding_box_percentage']['pressure'],
        info['bounding_box_bounds']['longitude'],
        info['bounding_box_percentage']['longitude'],
        info['bounding_box_bounds']['latitude'],
        info['bounding_box_percentage']['latitude']))


def make_friendly_storage_display(startpath = LOCAL_STORAGE_DIR, info_file = STORAGE_DISPLAY_INFO_FILE):
    """ prints the tree for the local_storage
        local_storage/
            NOAA/
                year
                    file
                year
                    file
            ERA5/
                year
                    month
                        file
                        file
                        file
                    month
                        file
                        file
                year..."""
    # load the content of the storage_info
    with open(info_file, 'r') as file:
        storage_info = json.load(file)
    # format info to string for display
    for data_source in storage_info:
        for file in storage_info['data_source']:
            pass



def get_dataset_bounds_and_size(data_source, dataset_data):
    """ Returns the bounds by dimension associated to a dataset loaded from a file. 
        Also returns the size of the dataset and percentage of the memory size of a complete file for this data source (as defined by corresponding layout)
        Input:
            data_source: str -> name of the source
            dataset_data: xarray.Dataset
        Output:
            bounds: dict
            size: dict -> two keys 'absolute' and 'relative', resp. dataset size and percentage of complete file """
    metadata = make_metadata_from_dataset(data_source, dataset_data)
    bounds = metadata['bounds']
    # convert bounds in compatible types for JSON serialization
    for dim in bounds:
        if dim != 'time':
            bounds[dim] = [float(x) for x in bounds[dim]]
    bounds['time'] = [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in bounds['time']]
    size = int(dataset_data.nbytes/(1024**2))
    size = {'absolute':size,'relative':min(100,int(100*size/datasets_def[data_source]['storage_file_total_size']()))}
    return bounds,size


def area(bound_compare, grid, coordinate):
    """ returns the grid for a full coverage of the area defined by bound_compare along a coordinate (longitude or latitude)
            uses the grid as an example for the step"""
    if coordinate == 'longitude':
        first = grid[coordinate][(grid[coordinate] >= bound_compare[coordinate][0]) & (grid[coordinate] <= 360)]
        second = grid[coordinate][(grid[coordinate] <= bound_compare[coordinate][1]) & (grid[coordinate] >= 0)]
        return len(first)+len(second)
    else:
        return len(grid[coordinate][(grid[coordinate] >= bound_compare[coordinate][0]) & (grid[coordinate] <= bound_compare[coordinate][1])])


def percentage_world_coverage(metadata, one_day_dimensions):
    """ returns the percentage of spatial coverage of the data given by metadata compared to the world, europe and france
            europe and france spatial boundaries are defined at the start of the file"""
    grid = metadata['grid']
    world_coverage = len(grid['longitude'])*len(grid['latitude'])/(len(one_day_dimensions['longitude'])*len(one_day_dimensions['latitude']))
    if area(bound_europe, one_day_dimensions, 'longitude')*area(bound_europe, one_day_dimensions, 'latitude') == 0:
        europe_coverage = 0
    else:
        europe_coverage = area(bound_europe, grid, 'longitude')*area(bound_europe, grid, 'latitude')/(area(bound_europe, one_day_dimensions, 'longitude')*area(bound_europe, one_day_dimensions, 'latitude'))
    if area(bound_france, one_day_dimensions, 'longitude')*area(bound_france, one_day_dimensions, 'latitude') == 0:
        france_coverage = 0
    else:
        france_coverage = area(bound_france, grid, 'longitude')*area(bound_france, grid, 'latitude')/(area(bound_france, one_day_dimensions, 'longitude')*area(bound_france, one_day_dimensions, 'latitude'))
    return world_coverage*100, europe_coverage*100, france_coverage*100

def update_interdependant_info(storage_info):
    """ Update for all file the infos that are affected by the change in one file 
        Input :
            storage_info : dict
        Output :
            None """
    for file in storage_info['NOAA']:
        storage_info['NOAA'][file]['size_percentage'] = round(storage_info['NOAA'][file]['size']/storage_info['global_info']['size']*100,5)
    for file in storage_info['ERA5']:
        storage_info['ERA5'][file]['size_percentage'] = round(storage_info['ERA5'][file]['size']/storage_info['global_info']['size']*100,5)

def add_one_file_info_to_storage_info(storage_info,data_source,file_path):
    """ Get the information for one file and put it in storage_info 
        Input:
            storage_info : dict -> supposed to already have dict values for all data source names keys
            data_source : str -> name of the data source
            file_path : str
        Output:
            None -> just modify the storage_info
    """
    file_name = os.path.basename(file_path).rstrip('.nc')
    if file_name not in storage_info[data_source]:
        storage_info[data_source][file_name] = {}
    # get infos
    convert_timestamp_in_datetime = datasets_def[data_source]['convert_time']
    one_day_dimensions = datasets_def[data_source]['one_day_dimensions']()
    with xr.open_dataset(file_path, chunks='auto', decode_times=False) as data:
        # bounds and size
        bounds,size = get_dataset_bounds_and_size(data_source,data)
        storage_info[data_source][file_name]['bounds'] = bounds
        storage_info[data_source][file_name]["size"] = size['absolute']
        storage_info[data_source][file_name]["completion_level"] = size['relative']
        storage_info["global_info"]["size"] += storage_info[data_source][file_name]["size"]
        # area cover percentage /!\ associated functions should be reviewed
        metadata = {'dataset': data_source, 'grid': {'longitude': data['longitude'], 'latitude': data['latitude']}}
        storage_info[data_source][file_name]["world"], storage_info[data_source][file_name]["europe"], storage_info[data_source][file_name]["france"] = percentage_world_coverage(metadata, one_day_dimensions)
    

def update_storage_display_for_files(file_paths,data_sources):
    """ Updates the storage_info for a list of files and associated data sources
        Input:
            file_paths : list -> list of paths
            data_sources : list -> list of data sources
        Output:
            None
    """
    print("Updating storage_info file")
    # load storage_info, create it if does not exist
    if os.path.exists(STORAGE_DISPLAY_INFO_FILE):
        with open(STORAGE_DISPLAY_INFO_FILE, 'r') as file:
            storage_info = json.load(file)
    else:
        storage_info = {"global_info": {"size": 0}, "NOAA": {}, "ERA5": {}}
    # update each file
    for file_path,data_source in zip(file_paths,data_sources):
        add_one_file_info_to_storage_info(storage_info,data_source,file_path)
    # update interdependant infos
    update_interdependant_info(storage_info)
    # sort the storage_info (not sure if useful)
    storage_info[data_source] = dict(sorted(storage_info[data_source].items(), key=lambda item: item[0]))
    # write the updated version of storage_info
    with open(STORAGE_DISPLAY_INFO_FILE, 'w') as file:
        json.dump(storage_info, file)
    print("storage_info updated")

def update_storage_display_one_file(filepath,data_source):
    """ Update the storage_info for only one file 
        Input:
            data_source : str
            filepath : str
        Output:
            None """
    update_storage_display_for_files(data_source,[filepath],[data_source])


def update_whole_storage_display(startpath=LOCAL_STORAGE_DIR):
    """ Update the storage_info for all files """
    file_paths = []
    data_sources = []
    for data_source in os.listdir(startpath): # iterate over dataset directories
        if not os.path.isdir(os.path.join(startpath,data_source)):
            continue
        for file in os.listdir(os.path.join(startpath,data_source)):
            file_path = os.path.join(startpath, data_source, file)
            if os.path.isfile(file_path): # only consider files (nc files)
                file_paths.append(file_path)
                data_sources.append(data_source)
    update_storage_display_for_files(file_paths,data_sources)


##### LEGACY ##############

def create_bounding_box(data, one_day_dimensions, convert_timestamp_in_datetime, file_name):
    bounding_box_bounds = {'time': [int(data.time.values[0]), int(data.time.values[-1])],
                           'latitude': [float(data.latitude.values[0]), float(data.latitude.values[-1])],
                           'pressure': [float(data.pressure.values[0]), float(data.pressure.values[-1])],
                           'longitude': find_longitude_bounds(data['longitude'].values)}
    bounding_box_percentage = {'time': find_percentage_bounding_box(data, one_day_dimensions, 'time', convert_timestamp_in_datetime, file_name),
                               'longitude': find_percentage_bounding_box(data, one_day_dimensions, 'longitude', convert_timestamp_in_datetime, file_name),
                               'latitude': find_percentage_bounding_box(data, one_day_dimensions, 'latitude', convert_timestamp_in_datetime, file_name),
                               'pressure': find_percentage_bounding_box(data, one_day_dimensions, 'pressure', convert_timestamp_in_datetime, file_name)}
    return {'bounds': bounding_box_bounds, 'percentage': bounding_box_percentage}


def find_percentage_bounding_box(dataset, one_day_dimensions, dim, convert_timestamp_in_datetime, file_name):
    file_name = file_name.split('.')[0]
    if dim == 'time':
        hours_span = dataset.time.values[-1] - dataset.time.values[0] + 1
        if hours_span == 0:
            return 100
        if file_name == 'NOAA':
            return len(dataset.time)/(int(hours_span*4/24)+1)*100
        elif file_name == 'ERA5':
            return len(dataset.time)/hours_span*100
    else:
        print(dim, dataset[dim].values[0], dataset[dim].values[-1])
        print([(dataset[dim].values[0]<=one_day_dimensions[dim]) & (dataset[dim].values[-1]>=one_day_dimensions[dim])])
        print(one_day_dimensions[dim][(dataset[dim].values[0]<=one_day_dimensions[dim]) & (dataset[dim].values[-1]>=one_day_dimensions[dim])])
        return len(dataset[dim])/len(one_day_dimensions[dim][(dataset[dim].values[0]<=one_day_dimensions[dim]) & (dataset[dim].values[-1]>=one_day_dimensions[dim])])*100


def list_files(startpath = LOCAL_STORAGE_DIR, info_file = STORAGE_DISPLAY_INFO_FILE):
    """ prints the tree for the local_storage
        local_storage/
            NOAA/
                year
                    file
                year
                    file
            ERA5/
                year
                    month
                        file
                        file
                        file
                    month
                        file
                        file
                year..."""
    # load the content of the storage_info in a variable
    with open(info_file, 'r') as file:
        storage_info = json.load(file)
    old_f = []
    # browse our directory
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        # creates the different indentation
        indent_folder = ' ' * 4 * level
        indent_year = ' ' * 4 * (level + 1)
        indent_month = ' ' * 4 * (level + 2)
        indent_day = ' ' * 4 * (level + 3)
        # print the name of the storage
        print('{}{}/'.format(indent_folder, os.path.basename(root)))
        # if we have files in the directory
        if files and files[0] != STORAGE_DISPLAY_INFO_FILE.split('/')[-1]:
            # sort the files using the key created in natural_keys()
            files.sort(key=natural_keys)
            # browse our files
            for f in files:
                # remove the .nc part of the file name
                f = f.rstrip('.nc')
                # split the file name by '.', ex. cur_f=['ERA5',2000,1,2]
                cur_f = f.split('.')
                # if we have a NOAA file, print the branch in the right way (year-file)
                if cur_f[0] == 'NOAA':
                    print('      #################### \n{}{}\n      ####################'.format(indent_year, cur_f[1]))
                    print_file_info(f, storage_info, indent_month)
                # if we have a ERA5 file, print the branch in the right way (year-month-file)
                if cur_f[0] == 'ERA5':
                    # if it's a new year, we print it
                    if not old_f or cur_f[1] != old_f[1]:
                        print('{}{}'.format(indent_year, cur_f[1]))
                    # if it's a new month, we print it
                    if not old_f or cur_f[2] != old_f[2] or cur_f[1] != old_f[1]:
                        print('       #####################\n{}{} {}\n       #####################'.format(indent_month, calendar.month_name[int(cur_f[2])], cur_f[1]))
                    print_file_info(f, storage_info, indent_day)
                    old_f = cur_f
        print('\n'*2)
        print('#############################################################')
        print('\n'*2)
