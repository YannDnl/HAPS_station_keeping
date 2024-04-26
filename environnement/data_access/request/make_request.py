import os.path

from dask.diagnostics import ProgressBar

from environnement.data_access.request_build.metadata import make_metadata_from_request_item
from environnement.data_access.request.concatenate import *
from environnement.data_access.config import *
from environnement.data_access.datasets.datasets_def import datasets_def
from environnement.data_access.storage_display import update_storage_display_one_file
from environnement.data_access.coordinates_system import find_longitude_bounds

def create_restricted_metadata(metadata, restricted_time_bound):
    """ create the time restricted metadata from metadata using the restricted time bounds corresponding to the type of dataset.
        As resulting data (intersection of data domain and restricted_time_bound) can be empty, return as first value a boolean set to True if restricted_metadata is NOT empty (success)"""
    restricted_metadata = {'dataset': metadata['dataset'],
                           'bounds': {},
                           'grid': {
                               'time': metadata['grid']['time'][(metadata['grid']['time'] >= restricted_time_bound[0]) &
                                                                (restricted_time_bound[-1] >= metadata['grid']['time'])],
                               'longitude': metadata['grid']['longitude'],
                               'latitude': metadata['grid']['latitude'],
                               'pressure': metadata['grid']['pressure']}}
    # check if not empty
    if len(restricted_metadata['grid']['time']) == 0:
        contains_data = False
        return contains_data, restricted_metadata
    contains_data = True
    restricted_metadata['bounds'] = {'time': [restricted_metadata['grid']['time'][0], restricted_metadata['grid']['time'][-1]],
              'longitude': metadata['bounds']['longitude'],
              'latitude': metadata['bounds']['latitude'],
              'pressure': metadata['bounds']['pressure']}
    return contains_data,restricted_metadata


def data_missing(restricted_metadata, file_path, echo=False):
    """ returns a missing_metadata corresponding to the metadata that is not present in the actual storage but is requested
            returns a boolean indicating if there are data missing or not"""
    are_data_missing = False
    missing_metadata = {'dataset': restricted_metadata['dataset'],
                    'bounds': {'time': [], 'longitude': [], 'latitude': [], 'pressure': []},
                    'grid': {'time': [], 'longitude': [], 'latitude': [], 'pressure': []},
                    'percentage': 100.0}
    # if there is no file in the storage, all the metadata is missing
    if not os.path.exists(file_path):
        return restricted_metadata, True
    # otherwise we open the dataset as aggregated_dataset
    with xr.open_dataset(file_path, decode_times=False) as agregated_dataset:
        # get the grid of the data missing and the data which are Nan
        grid_nan = find_nan_values(restricted_metadata, agregated_dataset)
    # merge those two grid into one sorted missing_metadata
    for key, value in grid_nan.items():
        missing_metadata['grid'][key] = np.sort(np.append(missing_metadata['grid'][key], grid_nan[key]))
    # get the bounds from the missing_metadata
    for name in missing_metadata['bounds']:
        # for the longitude we take the longest series of values without a hole in it
        if len(missing_metadata['grid'][name]) > 0:
            if name == 'longitude':
                missing_metadata['bounds']['longitude'] = find_longitude_bounds(missing_metadata['grid']['longitude'])
            else:
                missing_metadata['bounds'][name] = [missing_metadata['grid'][name][0], missing_metadata['grid'][name][-1]]
            are_data_missing = True
    # calculates the percentage of data_missing in the storage
    new_metadata_size = 1
    restricted_metadata_size = 1
    for name in restricted_metadata['grid']:
        new_metadata_size *= len(missing_metadata['grid'][name])
        restricted_metadata_size *= len(restricted_metadata['grid'][name])
    missing_metadata['percentage'] = new_metadata_size/restricted_metadata_size*100
    # echo missing data if flag is ON
    if echo:
        grid = missing_metadata['grid']
        print(f"Data missing in {file_path} :")
        print(f"- percentage missing : {missing_metadata['percentage']}%")
        # give details only if some data is missing
        if missing_metadata['percentage'] > 0.0:
            # if one data is missing, there is at least one missing coordinate for each dimension
            print(f"- first data missing : time {grid['time'][0]}, pressure {grid['pressure'][0]}, longitude {grid['longitude'][0]}, latitude {grid['latitude'][0]}")
            print(f"- bounds of missing : {missing_metadata['bounds']}")
        print("\n")
    return missing_metadata, are_data_missing


def find_nan_values(metadata, dataset):
    """ returns a grid of 4 dimensions of the values to fetch only regarding the coords in storage where a Nan value is spotted 
        /!\ Removed the chunking, because the lazy evaluation of combine first is too slow when a lot of small chunks
        If check of stored winds happen to be slow or to overflow memory, consider chunking with a chunk size of around 100 MB """
    grid = metadata['grid']
    convert_datetime_in_timestamp = datasets_def[metadata['dataset']]['convert_datetime']
    convert_timestamp_in_datetime = datasets_def[metadata['dataset']]['convert_time']
    nan_grid = {'time': [], 'pressure': [], 'longitude': [], 'latitude': []}
    shape = (len(grid['time']),len(grid['pressure']),len(grid['latitude']),len(grid['longitude']))
    # create a DataArray filled with True values of the size of metadata['grid']
    metadata_nan = xr.DataArray(np.ones(shape, dtype='float16'),
                        dims=("time", "pressure", "latitude", "longitude"),
                        coords={'time': [convert_datetime_in_timestamp(date) for date in grid['time']],
                                'pressure': grid['pressure'],
                                'latitude': grid['latitude'],
                                'longitude': grid['longitude']})
    # check for nan values in our variable for the metadata requested, nan values are indicated by a True and the rest is False
    dataset_is_nan = xr.DataArray(np.isnan(dataset.sel(time=np.intersect1d([convert_datetime_in_timestamp(date) for date in grid['time']], dataset['time']),
                                                       pressure=np.intersect1d(grid['pressure'], dataset['pressure']),
                                                       longitude=np.intersect1d(grid['longitude'], dataset['longitude']),
                                                       latitude=np.intersect1d(grid['latitude'], dataset['latitude']))['uwnd']).astype('float16'),
                                  dims=("time", "pressure", "latitude", "longitude"),
                                  coords={'time': np.intersect1d([convert_datetime_in_timestamp(date) for date in grid['time']], dataset['time']),
                                          'pressure': np.intersect1d(grid['pressure'], dataset['pressure']),
                                          'longitude': np.intersect1d(grid['longitude'], dataset['longitude']),
                                          'latitude': np.intersect1d(grid['latitude'], dataset['latitude'])})
    # combine our DataArray to get the mask of the whole requested array
    nan_mask = dataset_is_nan.combine_first(metadata_nan).astype('bool')
    # use np.where() to get the indices of the NaN values
    indices = [np.unique(elem) for elem in np.where(nan_mask)]
    for dim, index in zip(nan_mask.dims, indices):
        nan_grid[dim] = np.unique(np.concatenate((nan_grid[dim], nan_mask[dim].values[index])))
    if len(indices[0]) > 0:
        nan_grid['time'] = np.unique(np.append(nan_grid['time'], nan_mask['time'].values))
    # convert the indices to coordinates using the DataArray's dimensions
    nan_grid['time'] = [convert_timestamp_in_datetime(date) for date in nan_grid['time']]
    return nan_grid


def fetch_data_single_file(file_path, restricted_metadata):
    print(f"Fetching for file {file_path}...")
    # check the missing data in local_storage
    restricted_metadata, are_data_missing = data_missing(restricted_metadata, file_path)
    # stop here if no data missing
    if not are_data_missing:
        print("Data already in storage.")
        return
    # if data are missing, we can add them to the local_storage
    # if the dataset is too large (>3GB) we only fetch a subsets of it
    if datasets_def[restricted_metadata['dataset']]['grid_memory_size'](restricted_metadata) > MAX_API_REQUEST_SIZE:
        # perform too fetch_data_mf with half of the data taken along time dimension
        _,restricted_metadata_1 = create_restricted_metadata(restricted_metadata,[restricted_metadata['grid']['time'][0], restricted_metadata['grid']['time'][int(len(restricted_metadata['grid']['time']) / 2)]])
        _,restricted_metadata_2 = create_restricted_metadata(restricted_metadata,[restricted_metadata['grid']['time'][int(len(restricted_metadata['grid']['time']) / 2)], restricted_metadata['grid']['time'][int(len(restricted_metadata['grid']['time']))]])
        fetch_data_single_file(file_path,restricted_metadata_1)
        fetch_data_single_file(file_path,restricted_metadata_2)
        return
    # Otherwise, make the fetch
    fetched_data = datasets_def[restricted_metadata['dataset']]['API'](restricted_metadata)
    # If file already exists, we merge fetched data and existing data
    # 
    encoding = fetched_data['uwnd'].encoding
    if os.path.exists(file_path):
        with xr.open_dataset(file_path, chunks='auto', decode_times=False) as file_dataset:
            # since the whole file will be overwritten we concatenate both local_storage and data before overwriting
            fetched_data = fetched_data.combine_first(file_dataset)
            fetched_data = fetched_data.chunk('auto')
            fetched_data['uwnd'].encoding = encoding
            fetched_data['vwnd'].encoding = encoding
    fetched_data = fetched_data.sortby(['time','longitude','latitude','pressure'])
    # Write the result
    # if file already exists, first delete it as overwrite generate an error
    if os.path.exists(file_path):
        temp_file_path = file_path.rstrip('.nc') + '(1).nc'
        fetched_data.to_netcdf(temp_file_path, mode='w')
        print(file_path)
        os.remove(file_path)
        os.rename(temp_file_path, file_path)
    # add a progress bar to track the evolution of the writing
    with ProgressBar():
        print(f"Writing to {file_path}")
        fetched_data.to_netcdf(file_path, mode='w')
    print('\n\n\n')

    # /!\ STORAGE DISPLAY IN WORK
    # update_file_info(fetched_data, restricted_metadata, file_path)
    fetched_data.close()

    """
        fetched_data.to_netcdf(
            file_path.rstrip(os.path.basename(file_path)) + os.path.basename(file_path).rstrip(
                '.nc') + '(1).nc', mode='w')
    os.remove(file_path)
    os.rename(file_path.rstrip(os.path.basename(file_path)) + os.path.basename(file_path).rstrip(
        '.nc') + '(1).nc', file_path)"""



############ LEGACY #####################################################

def is_request_metadata_valid(metadata):
    """ check if data corresponding to request's
            metadata is contained in local storage """
    # check local storage data ...
    # return True / False
    with xr.open_dataset(LOCAL_STORAGE_DIR) as local_storage:
        for name in metadata['grid']:
            if not all(np.isin(metadata['grid'][name], local_storage.variables[name])):
                return False
    return True


def is_available(request_item=None, metadata=None):
    """ check if data from request_item or metadata in already in local storage
            if data already available, raise error"""
    # can be called either with a request item or direcly a metadata
    # check that one among request_item or metadata is not None, otherwise raise error

    if request_item is None and metadata is None:
        raise ValueError("Both request_item and metadata cannot be None")

    if metadata is None:
        metadata = make_metadata_from_request_item(request_item)
    # use is_request_metadata_valid function of local storage
    return is_request_metadata_valid(metadata)


def get_data(metadata):
    """ retrieve data of a given request from its metadata.
            metadata is assumed to have been validated before,
                otherwise raise error when data is found unavailable """

    if is_available(None, metadata):
        grid = metadata['grid']

        with xr.open_dataset(LOCAL_STORAGE_DIR, decode_times=False, engine='netcdf4') as dataset:
            data = xr.Dataset(dataset.sel(time=grid['time'],
                                          pressure=grid['pressure'],
                                          longitude=grid['longitude'],
                                          latitude=grid['latitude']))
        return data
    else:
        raise ValueError("requested data not in local_storage")


def fetch_data(metadata, data):
    """ add data to local storage
            if data already in local storage, raise error"""
    if is_available(None, metadata):
        raise ValueError("data already in local_storage")
    else:
        with xr.open_dataset(LOCAL_STORAGE_DIR, chunks='auto') as storage:
            concat = concat_all(data, storage)
    concat = concat.chunk('auto')
    # put as a convention that if no value, we have a Nan
    concat['uwnd'] = xr.where(concat['uwnd'] < -9900, np.nan, concat['uwnd'])
    concat['vwnd'] = xr.where(concat['vwnd'] < -9900, np.nan, concat['vwnd'])

    with ProgressBar():
        print(f"Writing to {LOCAL_STORAGE_DIR}")
        concat.to_netcdf(LOCAL_STORAGE_DIR + 'nc_test.nc')
    concat.close()


def fetch_data_zarr(metadata, data):
    if is_available(None, metadata):
        raise ValueError("data already in local_storage")

    with xr.open_zarr(LOCAL_STORAGE_DIR) as storage:
        concat = concat_all(data, storage)
    concat = concat.chunk('auto')

    for name_dim in data.dims:
        other_names = [name for name in data.dims if name != name_dim]
        with xr.open_zarr(LOCAL_STORAGE_DIR) as storage:
            diff = np.setdiff1d(concat[name_dim], storage[name_dim])
            same = [np.intersect1d(storage[name], concat[name]) for name in other_names]

        if len(diff) > 0:
            with ProgressBar():
                print(f"Writing to {LOCAL_STORAGE_DIR}")
                concat.sel({name_dim: diff,
                            other_names[0]: same[0],
                            other_names[1]: same[1],
                            other_names[2]: same[2]}).to_zarr(LOCAL_STORAGE_DIR, append_dim=name_dim)


def send_split_values(concat, diff_time, max_size):
    if concat.nbytes > max_size > concat.nbytes / len(concat.time):
        divider = int(concat.nbytes / max_size)
        print(divider)
        new_time = np.array_split(diff_time, divider)
        while not all(concat.sel(time=split_time).nbytes < max_size for split_time in new_time):
            divider += 1
            new_time = np.array_split(concat.time, divider)
        print(divider)

        for split_time in new_time:
            print(concat.sel(time=split_time).nbytes)
            print(concat.sel(time=split_time))
            with ProgressBar():
                concat.sel(time=split_time).to_zarr('stor.zarr', append_dim='time')
    elif concat.nbytes / len(concat.time) > max_size:
        raise ValueError('Size not possible')

def find_missing_data(metadata, dataset, get=False):
    """ returns a grid of 4 dimensions of the values to fetch only regarding the missing coords in storage """
    grid = metadata['grid']
    names = ['time', 'pressure', 'longitude', 'latitude']
    names_data_missing = []
    missing_grid = {'time': [], 'pressure': [], 'longitude': [], 'latitude': []}
    convert_timestamp_in_datetime = datasets_def[metadata['dataset']]['convert_time']
    dataset_dims = {'time': [convert_timestamp_in_datetime(int(date)) for date in dataset.time],
                    'longitude': dataset.longitude,
                    'latitude': dataset.latitude,
                    'pressure': dataset.pressure}
    # check if there are coordinates in metadata not present in storage
    for name in grid:
        if np.setdiff1d(grid[name], dataset_dims[name]).any():
            names_data_missing.append(name)
    if get:
        for name in names_data_missing:
            missing_grid[name] = np.setdiff1d(metadata['grid'][name], dataset_dims[name])
        return missing_grid
    if len(names_data_missing) == 0:
        return missing_grid
    elif len(names_data_missing) == 1:
        missing_grid[names_data_missing[0]] = np.append(missing_grid[names_data_missing[0]],
                                                    np.setdiff1d(grid[names_data_missing[0]],
                                                                 dataset_dims[names_data_missing[0]]))
        for name in np.setdiff1d(names, names_data_missing):
            missing_grid[name] = np.append(missing_grid[name], grid[name])
    else:
        for name in names_data_missing:
            missing_grid[name] = np.append(missing_grid[name], np.setdiff1d(grid[name], dataset_dims[name]))
            missing_grid[name] = np.append(missing_grid[name], np.intersect1d(grid[name], dataset_dims[name]))
            missing_grid[name] = sorted(missing_grid[name])
        for name in np.setdiff1d(names, names_data_missing):
            missing_grid[name] = np.append(missing_grid[name], grid[name])
    return missing_grid



