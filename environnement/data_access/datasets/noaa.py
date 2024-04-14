import os.path
import requests
import xarray as xr
import io
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from environnement.data_access.config import LOCAL_STORAGE_DIR


def convert_lon_lat(bound):
    return (bound +360)%360


def one_day_dimensions_of_NOAA():
    """returns the one day grid for the dataset, that is to say the example grid for a day"""
    grid = {
        'pressure': np.array([10,20,30,50,70,100,150,200,250,300,400,500,600,700,850,925,1000]),
        'latitude': np.arange(-90,90 + 2.5, 2.5),
        'longitude': np.arange(0,360, 2.5),
        'time': [0,6,12,18]
    }
    return grid


def create_url_noaa(metadata, var):
    """ create the noaa url for the requested data """

    N= str(metadata['bounds']['latitude'][1])
    W = str(metadata['bounds']['longitude'][0])
    E = str(metadata['bounds']['longitude'][1])
    S = str(metadata['bounds']['latitude'][0])

    start = metadata['bounds']['time'][0]
    end = metadata['bounds']['time'][1]

    if var == 'uwnd': ID = '2635'
    else: ID = '2633'
    # entête de base
    url = "https://www.psl.noaa.gov/cgi-bin/mddb2/plot.pl?doplot=0&varID="+ID+"&fileID=0&itype=0&variable="+var+"&levelType=Pressure%20Levels&level_units=millibar"
    # ajoute la pression
    for pressure in metadata['grid']['pressure'][::-1]:
        temp = "&level=" + str(float(pressure))
        url = url + temp

    #ajoute le temps
    url += "&timetype=4x&fileTimetype=4x&createAverage=1"
    url += "&year1=" + str(start.year) + "&month1=" + str(start.month) + "&day1=" + str(start.day) + "&hr1=" + str(start.hour) + "%20Z"
    url += "&year2=" + str(end.year) + "&month2=" + str(end.month) + "&day2=" + str(end.day) + "&hr2=" + str(end.hour)

    # ajoute emplacement
    url += "%20Z&region=" + "Custom"
    url += "&area_north=" + N + "&area_west=" + W + "&area_east=" + E + "&area_south=" + S
    url += "&centerLat=0.0&centerLon=270.0"

    return url


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

    with xr.open_dataset(file_like_object, chunks='auto', decode_times=False) as dataset:
        dataset = dataset.rename({'level': 'pressure', 'lon': 'longitude', 'lat': 'latitude'})
        dataset = dataset.assign_coords(longitude=(convert_lon_lat(dataset.longitude)))

        return dataset.sortby('time')


def fetch_wind_with_API_of_NOAA(metadata):
    """ fetch the requested data from database and return it """
    print("Fetching uwnd data...")
    url_uwnd= create_url_noaa(metadata, 'uwnd')
    dataset_uwnd = load_file(url_uwnd)

    print("Fetching vwnd data...")
    url_vwnd = create_url_noaa(metadata, 'vwnd')
    dataset_vwnd = load_file(url_vwnd)

    dataset = xr.merge([dataset_uwnd, dataset_vwnd])
    dataset_vwnd.close()
    dataset_uwnd.close()

    return dataset


def grid_memory_size_of_NOAA(metadata):
    """compute the memory size of values corresponding to metadata """
    """ the memory unit is MB """
    size = 1
    for name in metadata['grid']:
        size = size * len(metadata['grid'][name])
    memory_size = (4*size/1024/1024)*2
    return memory_size


def storage_time_split_NOAA(grid):
    """ /!\ WILL BE RELOCATED TO A LAYOUT FILE (with one argument : variable nbytes storage)
        returns the new_bounds associated with the given bound for NOAA datasets
            returns the name of the file associated with the new_bounds"""
    years = sorted(list(set(np.array([time.year for time in grid['time']]))))
    restricted_time_bounds = [[datetime(year, 1, 1, 0), datetime(year, 12, 31, 23)] for year in years]
    file_names = [os.path.join(LOCAL_STORAGE_DIR, 'NOAA', f"NOAA.{year}.nc") for year in years]
    return restricted_time_bounds, file_names

def storage_file_total_size_NOAA():
    """ /!\ WILL BE RELOCATED TO A LAYOUT FILE (with one argument : variable nbytes storage)
    -> should not be hardcoded but linked to storage_time_split as defined by the layout
        Returns total memory for one complete storage file """
    return grid_memory_size_of_NOAA({'grid':one_day_dimensions_of_NOAA()}) * 365


def convert_timestamp_in_datetime_NOAA(input_timestamp, origin_datetime = datetime.strptime("1800-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")):
    """ return a datetime equal to the origin datetime + hours equal to input_timestamp """
    input_timestamp = int(input_timestamp)
    return origin_datetime + timedelta(hours = input_timestamp)

def convert_datetime_in_timestamp_NOAA(input_datetime, origin_datetime = datetime.strptime("1800-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")):
    """ return a datetime equal to the origin datetime + hours equal to input_timestamp """
    return int((input_datetime - origin_datetime).total_seconds() // 3600)