from datetime import datetime
import matplotlib.pyplot as plt
from time import time
import sys

from wind_analysis.analysis_engine import analyse_wind
from wind_analysis.analysis_pipelines import month_analysis
from wind_analysis.report_generator import visualize_analysis
from wind_analysis.navigation_maps import make_interpolated_navigation_map,make_year_interpolated_navigation_map
from GUI.WindApp import WindApp

from data_access.request_build.requests_manager import fetch,get,delete,metadata
from data_access.request.echo_state import list_files, update_whole_cache

from CLI import *



#WindApp()
#pipeline_NOAA_NCEP_reanalysis_year()

"""
bounds = {'pressures':[300,50],'longitudes':[-5,5],'latitudes':[40,53]}
analysis = month_analysis(2022,9,bounds)
#print(analysis['minimum_wind_speed_avg'], analysis['minimum_wind_speed_hist'],analysis['pressure_where_min_speed_hist'],sep='\n')
visualize_analysis(analysis)"""

#print(get_wind_data()['metadata'])

request_item = {
    'dataset': 'ERA5',
    'memory_limit': 200*1000,
    'bounds': {
        'longitude': [-180,180],
        'latitude': [-90,90],
        'pressure':[850,1000],
        'time':[{'year':2018,'month':1,'day':1,'hour':0},{'year':2018,'month':1,'day':1,'hour':23}]
    },
    'subsampling':{
        'longitude':1,
        'latitude':1,
        'pressure':1,
        'hour':1,
        'month':1,
        'day':1
    }
}
#fetch(request_item)

"""
t0 = time()
analysis = analyse_wind(wind_data)
t1 = time()
print(f"Analysed wind in {t1-t0} seconds")
visualize_analysis(analysis)"""

#list_files()


request_item2 = {
    'dataset': 'NOAA',
    'memory_limit': 500*1000,
    'bounds': {
        'longitude': [-180,180],
        'latitude': [-90,90],
        'pressure':[0,1000],
        'time':[{'year':2022,'month':1,'day':1,'hour':0},{'year':2022,'month':12,'day':31,'hour':23}]
    },
    'subsampling':{
        'longitude':1,
        'latitude':1,
        'pressure':1,
        'hour':1,
        'month':1,
        'day':1
    }
}

if __name__ == "__main__":
    
    cli_prompt()
    
    #fetch(request_item)

    """
    wind_data = get(request_item)
    t0 = time()
    analysis = analyse_wind(wind_data)
    t1 = time()
    print(f"Analysed wind in {t1-t0} seconds")
    visualize_analysis(analysis)
    """

    #make_interpolated_navigation_map('ERA5', {"year":2019,"month":1,"day":1,"hour":3})


#wind_data = fetch(request_item2)

#make_year_interpolated_navigation_map('ERA5',2022,start_month=13,subsampling_hour=2)

#fetch(request_item)
#update_whole_cache()
