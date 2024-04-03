from environnement.data_access.request.make_request_mf import *
from environnement.data_access.config import DEFAULT_REQUEST_SIZE_LIMIT


def is_memory_size_below_limit(request_item=None, metadata=None):
    """ check that request does not exceed memory limit, either from request item or directly metadata """
    if metadata is None:
        metadata = make_metadata_from_request_item(request_item)

    memory_limit = request_item['memory_limit'] if 'memory_limit' in request_item else DEFAULT_REQUEST_SIZE_LIMIT
    # check that request does not exceed memory limit
    if datasets_def[metadata['dataset']]['grid_memory_size'](metadata) > float(memory_limit):
        return False
    return True

def fetch_and_get(request_item=None, metadata=None, skip_check = False):
    "Tries to fetch, then tries to get"
    fetch(request_item, metadata)
    return get(request_item, metadata, skip_check)


def fetch(request_item=None, metadata=None):
    """ fetch data from the right API following information from metadata or request,
            then add the fetched data in local storage"""
    # check that one among request_item or metadata is not None, otherwise raise error
    if metadata is None and request_item is None:
        raise ValueError("Both request_item and metadata cannot be None")
    if request_item is not None:
        sbs = request_item['subsampling']
        if request_item['dataset'] == 'NOAA':
            if sbs['month']!=1 or sbs['day']!=1 or sbs['hour']!=1 or sbs['longitude']!=1 or sbs['latitude']!=1:
                raise ValueError("The subsampling on time or coordinate have no effect on the fetch")
        if request_item['dataset'] == 'ERA5':
            if sbs['longitude']!=1 or sbs['latitude']!=1:
                raise ValueError("The subsampling on longitude and latitude have no effect while fetching")
    # compute metadata if argument is request_item
    if metadata is None:
        metadata = make_metadata_from_request_item(request_item)
    # check memory size of request, otherwise raise error
    if not is_memory_size_below_limit(request_item=request_item, metadata=metadata):
        raise ValueError("Memory of request above dataset's memory limit")
    # use fetch_data function of local storage to add new data
    print("Adding data to local_storage...")
    # fetch_data(metadata, fetched_data)
    fetch_data_mf(metadata)
    print("Data successfully added")


def get(request_item=None, metadata=None, skip_check = False):
    """ get data from local storage with information given by metadata or request,
            then returns the found data"""
    # check that one among request_item or metadata is not None, otherwise raise error
    if metadata is None and request_item is None:
        raise ValueError("Both request_item and metadata cannot be None")
    # compute metadata if argument is request_item
    if metadata is None:
        metadata = make_metadata_from_request_item(request_item)
    # check memory size of request, raise error if too high
    if not is_memory_size_below_limit(request_item=request_item, metadata=metadata):
        raise ValueError("Memory of request above dataset's memory limit")
    # make request
    wind_data = get_data_mf(metadata, skip_check)
    # add request item in case it is useful
    wind_data['request_item'] = request_item
    return wind_data


def delete(request_item=None, metadata=None):
    # check that one among request_item or metadata is not None, otherwise raise error
    if metadata is None and request_item is None:
        raise ValueError("Both request_item and metadata cannot be None")
    # compute metadata if argument is request_item
    if metadata is None:
        metadata = make_metadata_from_request_item(request_item)

    print("Deleting data")
    delete_data_mf(metadata)
    print("Data deleted")

def metadata(request_item,compute_bounds=True):
    """ simple request route to access metadata for a given request item 
        If you just want to get the grid for a specific dimension, use compute bounds = False and random value for other dimensions"""
    metadata = make_metadata_from_request_item(request_item,compute_bounds=compute_bounds)
    return metadata
