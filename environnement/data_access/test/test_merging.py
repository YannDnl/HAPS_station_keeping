
import xarray as xr
import time
import tracemalloc
import numpy as np
import gc

if __name__ == "__main__":
    tracemalloc.start()
    print("(current, peak) memory at start:")
    print(tracemalloc.get_traced_memory())

    # create the test data (each is 100 by 100 by 10 array of random floats)
    # Their A and B coordinates are completely matching. Their C coordinates are completely disjoint.

    data1 = np.random.rand(500, 200, 200, 20)
    da1 = xr.DataArray(
        data1,
        dims=("time", "pressure", "lon", "lat"),
        coords={
            "time": [i for i in range(500)],
            "pressure": [i for i in range(200)],
            "lon": [i for i in range(200)],
            "lat": [i for i in range(20)]},
    )
    da1.name = 'uwnd'
    da1 = xr.Dataset({'uwnd':da1, 'vwnd':da1})
    da1 = da1.chunk('auto')

    data2 = np.random.rand(500, 200, 200, 20)
    da2 = xr.DataArray(
        data2,
        dims=("time", "pressure", "lon", "lat"),
        coords={
            "time": [i+500 for i in range(500)],
            "pressure": [i for i in range(200)],
            "lon": [i for i in range(200)],
            "lat": [i for i in range(20)]},
    )
    da2.name = 'uwnd'
    da2 = xr.Dataset({'uwnd':da2, 'vwnd':da2})
    da2 = da2.chunk('auto')

    print("(current, peak) memory after creation of arrays to be combined:")
    before_merge_current, before_merge_peak = tracemalloc.get_traced_memory()
    print(tracemalloc.get_traced_memory())
    print(f"da1.nbytes = {da1.nbytes}")
    print(f"da2.nbytes = {da2.nbytes}")

    print(da1)
    # with xr.open_dataset("da_combined.nc") as imported:
    #     # da_combined = xr.merge([a_combined, imported])
    #     merged1 = imported['uwnd'].combine_first(da_combined['uwnd'])
    #     merged2 = imported['vwnd'].combine_first(da_combined['vwnd'])
    #
    # da_combined.close()
    # da_merged = xr.merge([merged1, merged2])

    # da_combined = da1.combine_first(da2)
    # da_combined = xr.merge([da1, da2])

    da_combined = xr.concat([da1, da2], 'time')
    # da_combined = xr.concat([da1, da2], ['time', 'pressure'])

    print(da_combined)
    print("(current, peak) memory after merging. You should observe that the peak memory usage is now much higher.")
    da1.close()
    da2.close()
    after_merge_current, after_merge_peak = tracemalloc.get_traced_memory()
    print(tracemalloc.get_traced_memory())
    print(f"da_combined.nbytes = {da_combined.nbytes}")
    # print(final)
    print(f"Consommation merge = {(after_merge_current-before_merge_current)/1024/1024}")
    print(f"Merge memory use = {after_merge_current/1024/1024}")
    print(f"Consommation peak = {after_merge_peak/1024/1024}")

    tracemalloc.stop()
    # da_combined.to_netcdf("da_combined.nc", mode='w')
