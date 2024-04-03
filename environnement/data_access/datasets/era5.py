import os.path
import cdsapi
import xarray as xr
import io
import numpy as np
import requests
from tqdm import tqdm
from datetime import datetime, timedelta, time
import urllib3
import math
from time import sleep

from environnement.data_access.config import LOCAL_STORAGE_DIR, ERA5_MAX_ITEMS_IN_REQUEST, ERA5_MAX_SLEEP_BETWEEN_ATTEMPTS
from environnement.data_access.request.concatenate import concat_all
from environnement.data_access.coordinates_system import find_longitude_bounds


def one_day_dimensions_of_ERA5():
    """returns the dimensions for one day of the dataset"""
    grid = {
        'pressure': np.array([
            1, 2, 3,
            5, 7, 10,
            20, 30, 50,
            70, 100, 125,
            150, 175, 200,
            225, 250, 300,
            350, 400, 450,
            500, 550, 600,
            650, 700, 750,
            775, 800, 825,
            850, 875, 900,
            925, 950, 975,
            1000,
        ]),
        'latitude': np.arange(-90, 90 + 0.25, 0.25),
        'longitude': np.arange(0, 360, 0.25),
        'time': np.arange(0,23 + 1, 1)
    }
    return grid

def convert_lon_lat_for_era5(bound):
    return -180+(bound+180)%360

def convert_lon_lat_from_era5(bound):
    return (bound +360)%360

def create_era5_url(metadata):
    """ get the url for the data from metadata using the cdsapi
            can be quite long due to the queueing system """
    # use the cdsapi for fetching era5 datasets
    c = cdsapi.Client()
    bounds = metadata['bounds']
    # get all the values from metadata that are useful for the request
    pressure = metadata['grid']['pressure']
    year = np.sort([date.strftime("%Y") for date in metadata['grid']['time']])
    month = np.sort([date.strftime("%m") for date in metadata['grid']['time']])
    day = np.sort([date.strftime("%d") for date in metadata['grid']['time']])
    time = np.sort([date.strftime("%H:%M") for date in metadata['grid']['time']])
    area = [bounds['latitude'][1], convert_lon_lat_for_era5(bounds['longitude'][0]), bounds['latitude'][0], convert_lon_lat_for_era5(bounds['longitude'][1])]
    # make the API request
    r = c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': ['u_component_of_wind', 'v_component_of_wind',],
            'pressure_level': pressure.tolist(),
            'year': list(set(year)),
            'month': list(set(month)),
            'day': list(set(day)),
            'time': list(set(time)),
            'area': area,
        })
    # r.location returns the url of the requested dataset
    return r.location


def load_file(url):
    """ load the file from an url and return it
        (added a progress bar to see download advancement) """

    response = requests.get(url, stream=True)
    response.raise_for_status() #raise error if request not successfull

    total_size = int(response.headers.get('content-length', 0))
    # create a file like object to write in the file from url
    file_like_object = io.BytesIO()
    
    # use a progress bar to watch the evolution of the downloading
    with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as progress_bar:
        # Iterate over the response content in chunks and write it to the BytesIO object
        for chunk in response.iter_content(chunk_size=8192):
            file_like_object.write(chunk)
            progress_bar.update(len(chunk))
    
    # Reset the file-like object's position to the start of the file
    file_like_object.seek(0)

    dataset = xr.open_dataset(file_like_object, chunks='auto', decode_times=False)
    response.close()

    return dataset


def secure_request_load(restricted_metadata):
    """ loop on the request while an error is catched.
        Sleeps between two request attempts, sleeping time multiplied by 2 between at each failure until max is reached.
        Input:
            restricted_metadata : dict
        Output:
            dataset : xarray.Dataset """
    success = False
    sleep_time = 1
    while not success:
        try:
            url = create_era5_url(restricted_metadata)
            dataset = load_file(url)
            success = True
        except Exception as e:
            print('An error has been raised by ERA5 request.')
            print(e)
            print('Retry...')
            sleep_time = min(ERA5_MAX_SLEEP_BETWEEN_ATTEMPTS,sleep_time*2)
            sleep(sleep_time)
    return dataset


""" cannot fetch more than 120 000 element (pressure * time) at once with era5 """
def fetch_wind_with_API_of_ERA5(metadata): # dataset=None pour tester de récupérer des petits dataset plutôt qu'un gros
    """ fetch the requested data from database and return it 
        Input: 
            metadata : dict
        Output:
            dataset_era5 : xarray.Dataset """
    urllib3.disable_warnings()
    grid = metadata['grid']
    bounds = metadata['bounds']
    # convert longitude bounds in [-180,180] format
    ERA5_longitude_bounds = find_longitude_bounds(metadata['grid']['longitude'],min_lon_value=-180)
    ERA5_longitudes = -180 + (grid['longitude'] + 180) % 180

    # divide in two requests if 180° is in the interval
    if ERA5_longitude_bounds[0] > ERA5_longitude_bounds[-1]:
        restricted_metadata_1 = {
            'grid': {'time': metadata['time'],'pressure': metadata['pressure'],
                     'longitude': [lon for lon in ERA5_longitudes if lon >= ERA5_longitude_bounds[0]],
                     'latitude': metadata['latitude']},
            'bounds': metadata['bounds']
        }
        restricted_metadata_2 = {
            'grid': {'time': metadata['time'],'pressure': metadata['pressure'],
                     'longitude': [lon for lon in ERA5_longitudes if lon <= ERA5_longitude_bounds[1]],
                     'latitude': metadata['latitude']},
            'bounds': metadata['bounds']
        }
        first_fetch = fetch_wind_with_API_of_ERA5(restricted_metadata_1)
        second_fetch = fetch_wind_with_API_of_ERA5(restricted_metadata_2)
        return xr.concat([first_fetch, second_fetch],'longitude')

    # there is a condition for the request made, time_len*pressure_len must be below 120000 elements
    # normally useless for request included in a single day
    time_chunks = math.ceil(len(grid['time'])*len(grid['pressure']) * 2 / ERA5_MAX_ITEMS_IN_REQUEST)
    if(time_chunks > 1):
        print("Data too large for a single ERA5 request, divided into", time_chunks, "datasets")

    # Load dataset by time chunk
    cut_metadata = {'bounds': metadata['bounds'], 'grid': {}}
    dataset_era5 = None
    for chunk_id in range(time_chunks):
        cut_metadata['grid'] = {'time': grid['time'][int(len(grid['time'])*(chunk_id/time_chunks)):int(len(grid['time'])*((chunk_id+1)/time_chunks))],
                                'longitude': grid['longitude'],
                                'latitude': grid['latitude'],
                                'pressure': grid['pressure']}

        # if it's not the first fetch, we concatenate the global fecth with the new small fetch
        if chunk_id != 0:
            restricted_dataset = secure_request_load(cut_metadata)
            dataset_era5 = xr.concat([dataset_era5,restricted_dataset],'time')
        else:
            dataset_era5 = secure_request_load(cut_metadata)

    # Change names and conventions of the fetched dataset to correspond those of our storage
    if 'level' not in dataset_era5:
        dataset_era5 = dataset_era5.assign_coords(level=[grid['pressure'][0]])
        dataset_era5['u'] = dataset_era5['u'].expand_dims('level')
        dataset_era5['v'] = dataset_era5['v'].expand_dims('level')
    dataset_era5 = dataset_era5.assign_coords(longitude=(convert_lon_lat_from_era5(dataset_era5.longitude)))
    dataset_era5 = dataset_era5.rename({'u': 'uwnd', 'v': 'vwnd', 'level': 'pressure'})
    dataset_era5 = dataset_era5.reindex(pressure=dataset_era5['pressure'][::-1])
    dataset_era5['pressure'] = dataset_era5['pressure'].astype('float32')
    dataset_era5 = dataset_era5.transpose('time', 'pressure', 'latitude', 'longitude')
    dataset_era5 = dataset_era5.sortby('time')
    dataset_era5 = dataset_era5.sortby('longitude')

    return dataset_era5


def grid_memory_size_of_ERA5(metadata):
    """compute the memory size of values corresponding to metadata """
    """ the memory unit is MB """
    size = 1
    for name in metadata['grid']:
        size = size * len(metadata['grid'][name])
    memory_size = (2*size/(1024**2))*2
    return memory_size


def storage_time_split_ERA5(grid):
    """ /!\ WILL BE RELOCATED TO A LAYOUT FILE 
        returns the restricted time bounds and corresponding file paths for a ERA5 grid """
    days_in_grid = sorted(list(set([dt.date() for dt in grid['time']])))
    restricted_time_bounds = []
    file_names = []
    for day in days_in_grid:
        if any((grid['time']>=datetime.combine(day, time(0))) & (grid['time']<=datetime.combine(day, time(11)))):
            restricted_time_bounds.append([datetime.combine(day, time(0)), datetime.combine(day, time(11))])
            file_names.append(os.path.join(LOCAL_STORAGE_DIR, 'ERA5',f"ERA5.{day.year}.{day.month}.{day.day}.{0}.nc"))
        if any((grid['time']>=datetime.combine(day, time(12))) & (grid['time']<=datetime.combine(day, time(23)))):
            restricted_time_bounds.append([datetime.combine(day, time(12)), datetime.combine(day, time(23))])
            file_names.append(os.path.join(LOCAL_STORAGE_DIR, 'ERA5',f"ERA5.{day.year}.{day.month}.{day.day}.{1}.nc"))
    return restricted_time_bounds, file_names

def storage_file_total_size_ERA5():
    """ /!\ WILL BE RELOCATED TO A LAYOUT FILE (with one argument : variable nbytes storage)
    -> should not be hardcoded but linked to storage_time_split as defined by the layout
        Returns total memory for one complete storage file """
    return grid_memory_size_of_ERA5({'grid':one_day_dimensions_of_ERA5()}) * 0.5

def convert_timestamp_in_datetime_ERA5(input_timestamp, origin_datetime = datetime.strptime("1900-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")):
    """ return a datetime equal to the origin datetime + hours equal to input_timestamp """
    input_timestamp = int(input_timestamp)
    return origin_datetime + timedelta(hours=input_timestamp)

def convert_datetime_in_timestamp_ERA5(input_datetime, origin_datetime = datetime.strptime("1900-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")):
    """ return a datetime equal to the origin datetime + hours equal to input_timestamp """
    return int((input_datetime - origin_datetime).total_seconds() // 3600)
