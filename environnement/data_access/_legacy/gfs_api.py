#!/usr/bin/env python
import datetime as dt
import warnings
import xarray as xr
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_access.config import GFS_DIR, GFS_API_FILE

GFS_BASE = "https://nomads.ncep.noaa.gov/dods"


def fetch_gfs_wind(
    date: dt.date = None,
    run: int = 0,
    hour: int = None,
    res: str = "0p25",
    step: str = "1hr",
):

    date = date if date is not None else dt.date.today()

    variables = ["ugrd10m", "vgrd10m"]

    date_str = date.strftime("%Y%m%d")
    url = f"{GFS_BASE}/gfs_{res}_{step}/gfs{date_str}/gfs_{res}_{step}_{run:02d}z"

    print(url)

    with warnings.catch_warnings():
        # xarray/coding/times.py:119: SerializationWarning: Ambiguous reference date string
        warnings.filterwarnings(
            "ignore",
            category=xr.SerializationWarning,
            module=r"xarray",
        )
        with xr.open_dataset(url) as dataset:
            # We use "nearest" in case of small precision problems
            if hour is None:
                fout = GFS_API_FILE
            else:
                time = dt.time(hour=hour)
                """
                dataset = ds[varlist].sel(
                    time=dt.datetime.combine(date, time), method="nearest"
                )"""
                dataset = dataset.sel(time=dt.datetime.combine(date, time), method="nearest")
                #fout = os.path.join(GFS_DIR,f"{date_str}_{run:02}_{hour:02}.nc")
                fout = GFS_API_FILE
            dataset.to_netcdf(fout)


def get_gfs_wind(date: dt.datetime = None, hour: int = 0, run: int = 0):
    dataset = xr.open_dataset(GFS_API_FILE)
    return dataset
    
fetch_gfs_wind(hour=0)
#print(get_gfs_wind())


