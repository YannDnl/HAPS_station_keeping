import numpy as np
import xarray as xr


def concat_all(add, storage):
    """ returns the merge of add and storage with a memory efficient method
            if add has one coordinate with nothing in common with storage -> concatenate on this coordinate
                otherwise, we take every coordinate of add which values are not exactly the same as in storage
                        we concatenate along those coordinates having selected the correct parts of storage and add """

    is_common, name_dim, diff = check_common_coordinates(add, storage)
    # if the coordinates are the same for storage and data
    if all(len(diff[name]) == 0 for name in diff) and is_common:
        return add
    # if one dimension has nothing in common with the storage, concatenate along this dimension
    if not is_common:
        concat = xr.concat([add, storage], dim=name_dim)
    else:
        check = 0
        # we go through the different coordinates, checking if there are some with different values than storage
        if len(diff['time']) > 0:
            # we concatenate along the time coordinate with selecting data from add not present in storage
            concat = concat_dim(add, storage, 'time', diff)
            check += 1
        if len(diff['longitude']) > 0:
            # if we haven't concatenated yet, make the first concatenation
            if check == 0:
                concat = concat_dim(add, storage, 'longitude', diff)
                check += 1
            else:
                data_lon_add = add.sel(longitude=np.setdiff1d(add['longitude'], storage['longitude']))
                storage_common = np.intersect1d(storage['longitude'], concat['longitude'])
                concat = xr.concat([concat.sel(longitude=storage_common), data_lon_add], dim='longitude')
        if len(diff['latitude']) > 0:
            if check == 0:
                concat = concat_dim(add, storage, 'latitude', diff)
                check += 1
            else:
                data_lat_add = add.sel(latitude=np.setdiff1d(add['latitude'], storage['latitude']))
                storage_common = np.intersect1d(storage['latitude'], concat['latitude'])
                concat = xr.concat([concat.sel(latitude=storage_common), data_lat_add], dim='latitude')
        if len(diff['pressure']) > 0:
            if check == 0:
                concat = concat_dim(add, storage, 'pressure', diff)
                check += 1
            else:
                data_pressure_add = add.sel(pressure=np.setdiff1d(add['pressure'], storage['pressure']))
                storage_common = np.intersect1d(storage['pressure'], concat['pressure'])
                concat = xr.concat([concat.sel(pressure=storage_common), data_pressure_add], dim='pressure')
    concat = concat.sortby('time')
    concat = concat.sortby('longitude')
    concat = concat.sortby('latitude')
    concat = concat.sortby('pressure')
    return concat


def check_common_coordinates(add, storage):
    """ Checks if the datasets have common coordinates.
        - separating_dim is a dimension with nothing in common if exists, else None
        - new_coordinates is a dict with coordinates of add that are not in storage for each dimension """
    new_coordinates = {}
    separating_dim = None
    has_common_coordinates = True
    for name_dim in add.dims:
        diff = np.setdiff1d(add[name_dim], storage[name_dim])
        new_coordinates[name_dim] = diff
        if len(diff) == len(add[name_dim]):
            has_common_coordinates = False
            separating_dim = name_dim
    return has_common_coordinates, separating_dim, new_coordinates


def concat_dim(add, storage, dim, diff):
    """ returns the concatenation of storage and data along dim coordinate selecting data with diff"""
    data = add.sel({dim: diff[dim]})
    return xr.concat([storage, data], dim=dim)