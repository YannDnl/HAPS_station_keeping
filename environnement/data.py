import datetime
import numpy as np

from environnement.data_access.request_build.requests_manager import fetch,get,delete,metadata

jours = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

request_item = {
            'dataset': "NOAA",
            'memory_limit': 200*1000,
            'bounds': {
                'latitude': [],
                'longitude': [],
                'pressure':[],
                'time': []
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

def back_time(time):
    time['hour'] -= 6
    if(time['hour'] < 0):
        time['hour'] = time['hour']%24
        time['day'] -= 1
        if(time['day'] < 1):
            time['month'] -= 1
            limite = jours[time['month'] - 1]
            if(time['month'] == 2 and time['year']%4 == 0):
                limite += 1
            time['day'] = limite
            if(time['month'] < 1):
                time['month'] = 12
                time['year'] -= 1

def request_vent(request_item):
    fetch(request_item)
    wind_data = get(request_item, skip_check=True)
    return wind_data['data']

def get_data(start_date):
    start_time = datetime.datetime(year = start_date['year'], month = start_date['month'], day = start_date['day'], hour = start_date['hour'])
    end_time = start_date
    end_time['year'] += 1
    back_time(end_time)
    end_time = datetime.datetime(year = end_time['year'], month = end_time['month'], day = end_time['day'], hour = end_time['hour'])
    request_item['bounds']['time'] = np.array([start_time, end_time])
    request_item['bounds']['latitude'] = np.array([-90 + i * 2.5 for i in range(73)])  #de -90 à 90 avec un pas de 2.5
    request_item['bounds']['longitude'] = np.array([i * 2.5 for i in range(144)])      #de 0 à 357.5 avec un pas de 2.5
    request_item['bounds']['pressure'] = np.array([10., 20., 30., 50., 70., 100., 150., 200., 250., 300., 400., 500., 600., 700., 850., 925., 1000.])
    return request_vent(request_item)

#data_loader
#