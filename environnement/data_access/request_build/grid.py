import calendar
import numpy as np
from datetime import datetime, timedelta



def create_grid_any(bounds, subsampling, one_day_dimensions, name):
    """ returns a np.array of the values between lower bound and upper bound using the subsampling
            and the one_day_dimensions, can adapt to longitude, latitude and pressure with the name input """
    one_day_dimensions_sbs = one_day_dimensions[name][::subsampling]
    if isinstance(subsampling, list):
        grid = np.array([get_closest_value(i, one_day_dimensions[name]) for i in subsampling if i >= bounds[0]])
    else:
        if len(bounds) == 1:
            return np.array([get_closest_value(bounds[0], one_day_dimensions[name])])
        if (name == 'longitude') and (bounds[0] > bounds[-1]):  # handle the longitude cycle (0,360°)
            grid = np.append(create_grid_any([bounds[0], 360], subsampling, one_day_dimensions, name),
                             create_grid_any([0, bounds[-1]], subsampling, one_day_dimensions, name))
            return grid
        index_min = get_closest_index_sup(bounds[0], one_day_dimensions_sbs)
        index_max = get_closest_index_inf(bounds[-1], one_day_dimensions_sbs)
        grid = one_day_dimensions_sbs[index_min:index_max+1]
    grid.sort()
    return grid


def create_grid_date(bounds, month_sbs, day_sbs, hour_sbs, one_day_dimensions):
    """ returns a np.array of the values between lower bound and upper bound using the subsampling,
            the output is in timestamp format"""
    year0, year1 = bounds[0].year, bounds[-1].year
    years = np.arange(year0,year1+1)
    months = np.arange(1,13,month_sbs)
    days = np.arange(1,32,day_sbs)
    hours = np.arange(0,24,hour_sbs)
    times = []
    for year in years:
        for month in months:
            for day in days:
                for hour in hours:
                    try:
                        time = datetime(year=year,month=month,day=day,hour=hour)
                    except ValueError: # if day does not exist
                        continue
                    if time.hour in one_day_dimensions['time'] and time >= bounds[0] and time <= bounds[-1]:
                        times.append(time)
    return np.array(times)

def get_closest_value(number, tab):
    """ returns number's closest value in tab  """
    closest_value = min(tab, key=lambda x: abs(x-number))
    return closest_value


def get_closest_index_sup(value, tab):
    """ returns the index of the upper closest value of value from tab """
    closest_index = None
    closest_difference = float('inf')
    for i, number in enumerate(tab):
        if number >= value and number - value < closest_difference:
            closest_difference = number - value
            closest_index = i
    if closest_index is None:
        return get_closest_index_inf(value, tab)
    return closest_index


def get_closest_index_inf(value, tab):
    """ returns the index of the lower closest value of value from tab """
    closest_index = None
    closest_difference = float('inf')
    for i, number in enumerate(tab):
        if number <= value and value - number < closest_difference:
            closest_difference = value - number
            closest_index = i
    if closest_index is None:
        return get_closest_index_sup(value, tab)
    return closest_index

