import pickle
import os
from time import time
from datetime import datetime

from environnement.data_access.config import WORK_DATA_DIR, get_month_wind_data_pickle_file, get_year_metadata_pickle_file

""" This file must change of structure soon.
    The access to wind data should be made with a function whose args are:
        - the source data (local files, remote server...)
        - the desired dataset
        - the desired area
        - the desired pressure range
        - a desired time range or a minimum time range, with or without extension if enough RAM 
    Metadata should include:
        - a variable (as a dict) domain_of_definition that explicits the time,pressure,lon,lat limits. Lon limits are given west-to-east. 
        - data grid points along the different dimensions, consequence of dataset resolution """

def get_wind_data(year=2022,month=1,pressure_range=[1100,0]):
    """ Return one month of wind data in given pressure range.
        Pressure range is a list of two elements : upper and lower bounds (decreasing order)
        The result is a dict with two keys:
            - data: wind corresponding to the period, organized as a time-pressure dict
            - metadata: a dict containg informations that give context about the loaded data. Its keys are:
                - values included in the returned data for the different grid axis : 'times','pressures','longitudes','latitudes', and also 'LON interval' and 'LAT interval'.
                  These values are produced along with data loading, the remaining is enrichment of these.
                - 'domain_of_definition' : dict that summarizes the data space by expliciting bounds for the different dimensions (bounds are given in 'natural' order, e.g time in increasing order, pressure in decreasing order)
                - 'request_info' : dict that stores the bounds given for each of the dimensions in the request
    """
    print(f"Loading wind work data for year {year} and month {month}...")
    result = {}
    t1 = time()
    # loading data
    fn = get_month_wind_data_pickle_file(year,month)
    with open(fn,'rb') as pickle_in:
        wind_data = pickle.load(pickle_in)
    # loading grid values metadata
    result['metadata'] = {}
    fn = get_year_metadata_pickle_file(year)
    with open(fn,'rb') as pickle_in:
        result['metadata']['grid'] = pickle.load(pickle_in)
    # /!\ FILTER TIMES METADATA /!\ -> should not be done here
    result['metadata']['grid']['times'] = [t for t in result['metadata']['grid']['times'] if t.year == year and t.month == month]
    # /!\ FILTER PRESSURES /!\ -> should not be done here
    result['metadata']['grid']['pressures'] = [p for p in result['metadata']['grid']['pressures'] if p <= pressure_range[0] and p >= pressure_range[1]]
    wind_data = {t:{p:by_time[p] for p in result['metadata']['grid']['pressures']} for t,by_time in wind_data.items()}
    result['data'] = wind_data
    # domain of definition
    domain_of_definition = {
        'time':[result['metadata']['grid']['times'][0],result['metadata']['grid']['times'][-1]],
        'pressure':[result['metadata']['grid']['pressures'][0],result['metadata']['grid']['pressures'][-1]],
        'longitude':[0,360],
        'latitude':[-90,90]
    }
    result['metadata']['domain_of_definition'] = domain_of_definition
    # request info
    request_info = {
        'times': [datetime(year=year,month=month,day=1),datetime(year=year+(month==12),month=1+(month)%12,day=1)],
        'pressures': pressure_range,
        'longitude' : [0,360],
        'latitude': [-90,90]
    }
    result['request_info'] = request_info
    t2 = time()
    print(f"Wind work data loaded in {t2-t1} seconds.")
    return result


