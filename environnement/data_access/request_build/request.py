from datetime import datetime


def create_request_item():
    """ creates the request_item with the chosen convention """

    request_item = {
        'memory_limit': input("Memory limit in MB: "), 'dataset': input("Dataset: "),
        'bounds': {
            'time': input("Time bounds (lower and upper) in format Y-M-D-H: "),
            'pressure': input("Pressure bounds (lower and upper) in hPa, separated by space: "),
            'longitudes': input("Longitudes bounds (lower and upper) separated by space: "),
            'latitudes': input("Latitudes bounds (lower and upper): "),
        },
        'subsampling': {
            'month': input("Month subsampling factor / list of months (integers in [1,12]): "),
            'day': input("Day subsampling factor / list of days (integers in [1,31]): "),
            'time': input("Time subsampling factor / list of hours (integers in [0,23]): "),
            'pressure': input("Pressure subsampling factor / list of pressures (float values): "),
            'longitude': input("Longitude subsampling factor / step between longitudes (float value): "),
            'latitude': input("Latitude subsampling factor / step between latitudes (float value): ")
        }
    }
    return request_item


def create_prefilled_request():
    start = {'year': 2002,
             'month': 1,
             'day': 1,
             'hour': 3}
    end = {'year': 2002,
           'month': 12,
           'day': 1,
           'hour': 11}


    request_item = {
      "dataset": "ERA5",
        "memory_limit": 1500,
      "bounds": {
        "time": [start,end],  #soit format datetime python soit dictionnaire Y/M/D/H
        "pressure": [100,500],
        'longitude': [0.25,7],
        'latitude': [-90,90]
      },
      "subsampling": {
        "day": 3,
        "month": 2,
        "hour": 2,
        "pressure": 1,
        "longitude": 2,
        "latitude": 1
      },
    "fill_pattern":[
        {
        'dimension': 'time',
        'filling_rule': 'forward'
        },
        {
            'dimension': 'longitude',
            'filling_rule': 'forward'
        }
    ]
    }
    return request_item


