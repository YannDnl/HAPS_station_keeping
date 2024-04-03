import datetime

import numpy as np
from environnement.data_access.datasets.datasets_def import datasets_def
from environnement.data_access.request_build.metadata import make_metadata_from_request_item
from environnement.data_access.request_build.request import create_prefilled_request
from environnement.data_access.request_build.requests_manager import get, fetch, delete_data_mf
from environnement.data_access.storage_display import list_files, update_whole_storage_display
from environnement.data_access.request.make_request_mf import data_missing, is_available_mf
from environnement.data_access.datasets.era5 import fetch_wind_with_API_of_ERA5
from environnement.data_access.config import LOCAL_STORAGE_DIR
import xarray as xr
import os

if __name__ == '__main__':
    request = create_prefilled_request()
    list_files()
        # print(data_to_fetch(metadata, '/media/luca/Seagate Portable Drive/Stratolia/local_storage/ERA5/ERA5.2005.1.7.nc'))