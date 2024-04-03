import datetime

from environnement.data_access.request_build.requests_manager import fetch,get,delete,metadata

request_item = {
            'dataset': "NOAA",
            'memory_limit': 200*1000,
            'bounds': {
                'latitude': [0, 0],
                'longitude': [0, 0],
                'pressure':[10,10],
                'time': [datetime.datetime(year=2020,month=1,day=1,hour=0),datetime.datetime(year=2020,month=1,day=1,hour=5)]
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

def request_vent(request_item):
    fetch(request_item)
    wind_data = get(request_item, skip_check=True)
    return wind_data['data']

#data_loader
#