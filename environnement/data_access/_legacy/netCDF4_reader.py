import netCDF4 as nc
import os
import numpy as np

def load_netCDF4_dataset(filename):
    data = nc.Dataset(filename)
    return data

def display_netCDF4_dataset(filename):
    data = load_netCDF4_dataset(filename)
    print(data)

def display_var_of_netCDF4_dataset(filename, var_name):
    data = load_netCDF4_dataset(filename)
    descr, arr = data[var_name], data[var_name][:]
    print(descr)
    #print(arr)

def load_var_of_netCDF4_dataset(filename,var_name):
    data = load_netCDF4_dataset(filename)
    arr = np.ma.getdata(data[var_name][:])
    return arr


#display_netCDF4_dataset(os.path.join('data','raw','uwnd.2020.nc'))
#display_netCDF4_dataset(os.path.join('data','raw','GFS','test.nc'))
#display_var_of_netCDF4_dataset(os.path.join('data','raw','GFS','test.nc'),'ugrd10m')



