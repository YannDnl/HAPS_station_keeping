import time
import xarray as xr
import numpy as np
from environnement.data_access.request.make_request import data_missing
from environnement.data_access.datasets.datasets_def import datasets_def
from environnement.data_access.config import *
from environnement.data_access.request_build.metadata import make_metadata_from_request_item
from environnement.data_access.storage_display import update_storage_display_one_file
from environnement.data_access.request.make_request import fetch_data_single_file, create_restricted_metadata
from environnement.data_access.request_build.metadata import make_metadata_from_dataset

""" data_mf uses a multiple file data storage, each datasets are stored using a time span specific to the type of dataset (NOAA, ERA5...)"""


def fetch_data_mf(metadata):
    """ add data to local storage, local storage is made of many files,
            the code takes a bit of data from the data to add and add it in the correct file
                it repeats this until the whole data is added"""
    restricted_time_bounds, file_paths = datasets_def[metadata['dataset']]['storage_time_split'](metadata['grid'])
    # go through the bounds and filepaths, for each of them we add the data in the correct file in local_storage
    for restricted_time_bound, file_path in zip(restricted_time_bounds, file_paths):
        # create a restricted_metadata useful for fetching large datasets
        # this restricted_metadata is the same as metadata except that the time coordinates are limited by the restricted_time_bounds
        contains_data, restricted_metadata = create_restricted_metadata(metadata, restricted_time_bound)
        if contains_data:
            fetch_data_single_file(file_path, restricted_metadata)
        
def get_data_mf(metadata, skip_check = False):
    """ retrieve data of a given request from its metadata.
            metadata is assumed to have been validated before,
                otherwise raise error when data is found unavailable """

    restricted_time_bounds, file_paths = datasets_def[metadata['dataset']]['storage_time_split'](metadata['grid'])

    if not skip_check:
        # get the type of dateset (NOAA/ERA5) and we find the correct filepaths to open thanks to the metadata
        any_data_missing = False
        any_file_missing = not all(os.path.exists(file_path) for file_path in file_paths)
        # if file that should contain data not in local_storage, raise error
        if any_file_missing:
            files_missing = [file_path for file_path in file_paths if not os.path.exists(file_path)]
            error_msg = "Some files that should host queried data are not created :\n" + "\n".join(
                [f"- {file_path}" for file_path in files_missing])
            raise ValueError(error_msg)
        # if data is missing in at least one file, raise error
        for restricted_time_bound, file_path in zip(restricted_time_bounds, file_paths):
            contains_data, restricted_metadata = create_restricted_metadata(metadata, restricted_time_bound)
            if not contains_data: # ignore if no intersection with this file
                continue
            print(f'Check if data are missing in {file_path}')
            metadata_missing, are_missing = data_missing(restricted_metadata, file_path, echo=True)
            if are_missing:
                any_data_missing = True
        if any_data_missing:
            raise ValueError('Some data are not in local storage')

    # else retrieve data
    convert_datetime_in_timestamp = datasets_def[metadata['dataset']]['convert_datetime']
    with xr.open_mfdataset(file_paths, chunks='auto', decode_times=False) as agregated_dataset:
        # select the data to get from request according to metadata
        data = agregated_dataset.sel(time=[convert_datetime_in_timestamp(date) for date in metadata['grid']['time']],
                                     pressure=metadata['grid']['pressure'],
                                     longitude=metadata['grid']['longitude'],
                                     latitude=metadata['grid']['latitude'])
        #data = data.transpose(*WIND_DATA_DIM_ORDER) # put dimensions in the chosen order
        u = np.expand_dims(data.uwnd.astype(DATA_TYPE_GET).values, axis=-1)
        v = np.expand_dims(data.vwnd.astype(DATA_TYPE_GET).values, axis=-1)
        wind_array = np.concatenate([u, v], axis=-1)
        wind_array = np.swapaxes(wind_array,2,3) # longitude-latitude order
        # recompute metadata based on loaded data
        metadata = make_metadata_from_dataset(metadata['dataset'], data)
    wind_data = {
        'metadata': metadata,
        'data': wind_array
    }
    return wind_data
        
    
def delete_data_mf(metadata):
    """ Delete data given by metadata from request and update the cache file"""
    # get the type of dateset (NOAA/ERA5) and we find the correct filepaths to open thanks to the metadata
    file_paths = datasets_def[metadata['dataset']]['storage_time_split'](metadata['grid']['time'])[1]
    # if the data is in the storage we continue
    for file_path in file_paths:
        if os.path.exists(file_path):
            # open local_storage
            for file_path in file_paths:
                with xr.open_dataset(file_path, chunks='auto', decode_times=False) as file_dataset:
                    # select the data to keep from request according to metadata
                    data = file_dataset.sel(
                        time=np.setdiff1d([datasets_def[metadata['dataset']]['convert_time'](int(date)) for
                                           date in file_dataset['time']], metadata['grid']['time']),
                        pressure=np.setdiff1d(file_dataset.pressure, metadata['grid']['pressure']),
                        longitude=np.setdiff1d(file_dataset.longitude, metadata['grid']['longitude']),
                        latitude=np.setdiff1d(file_dataset.latitude, metadata['grid']['latitude']))
                file_dataset.close()
                if data.nbytes > 0:
                    # rewrite the data we wanted to keep
                    data.to_netcdf(file_path, mode='w')
                else:
                    os.remove(file_path)
                # /!\ STORAGE DISPLAY IN WORK
                # update_file_info(data, metadata, file_path)
        else:
            raise ValueError("requested data not in local_storage")


####### LEGACY ####################

def is_available_mf(request_item=None, metadata=None):
    """ check if data corresponding to request's
            metadata is contained in local storage
            returns True if the data is present, False otherwise"""
    if request_item is None and metadata is None:
        raise ValueError("Both request_item and metadata cannot be None")

    if metadata is None:
        metadata = make_metadata_from_request_item(request_item)

    # get the type of dateset (NOAA/ERA5) and we find the correct filepaths to open thanks to the metadata
    data_type = metadata['dataset']
    file_paths = datasets_def[data_type]['storage_time_split'](metadata['grid']['time'])[1]
    grid = metadata['grid']
    convert_datetime_in_timestamp = datasets_def[data_type]['convert_datetime']
    percentage = {}
    # if one file doesn't exist, the data is not in the storage
    if not all(os.path.exists(file_path) for file_path in file_paths):
        return False
    # open all files which path is in filepaths
    with xr.open_mfdataset(file_paths, chunks='auto', decode_times=False) as agregated_dataset:
        for name in metadata['grid']:
            # check if data from metadata are present in the local_storage
            # if we check for the time we have to decode the values from request in the correct format
            if name == 'time':
                percentage[name] = 100*np.count_nonzero(np.isin(grid[name],[datasets_def[data_type]['convert_time'](int(date)) for date in agregated_dataset['time']]))/len(grid[name])
            else:
                percentage[name] = 100*np.count_nonzero(np.isin(metadata['grid'][name], agregated_dataset[name]))/len(grid[name])
        are_all_dim_values_in_dataset = all(dim_per == 100 for dim_per in percentage.values())
        return are_all_dim_values_in_dataset, percentage
        """
        nan_mask = np.isnan(local_storage.sel(
            time=np.intersect1d([convert_datetime_in_timestamp(date) for date in grid['time']], local_storage['time']),
            pressure=np.intersect1d(grid['pressure'], local_storage['pressure']),
            longitude=np.intersect1d(grid['longitude'], local_storage['longitude']),
            latitude=np.intersect1d(grid['latitude'], local_storage['latitude']))['uwnd'])
        indices = np.where(nan_mask)
        if all(index == [] for index in indices):
            return False
        """