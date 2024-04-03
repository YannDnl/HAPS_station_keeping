from environnement.data_access.request_build.request import create_prefilled_request
from environnement.data_access.request_build.requests_manager import fetch, get
from time import time
import tracemalloc
from environnement.data_access.request.make_request_mf import *



def test_accessing_big():
    tracemalloc.start()
    print("At start", tracemalloc.get_tracemalloc_memory())
    start0 = time()
    data = xr.open_zarr(LOCAL_STORAGE_WRITE+'add.zarr')
    print('Taille data ajouté :', data.nbytes/(1024**3))
    end0 = time()
    print("After opening add data", tracemalloc.get_tracemalloc_memory())
    start1 = time()
    with xr.open_zarr(LOCAL_STORAGE_DIR, chunks='auto') as storage:
        storage = storage.sortby('time')
        storage = storage.sel(time=slice(0, 256297))
        print("After opening storage", tracemalloc.get_tracemalloc_memory())
        end1 = time()
        new_storage = storage.combine_first(data)
        print('After concatenation', tracemalloc.get_tracemalloc_memory())
        end2 = time()
        new_data = storage.sel(time=slice(0,100000), lon=slice(0,99), lat=slice(0,99), level=slice(0,29))
        end3 = time()

        print('After selection', tracemalloc.get_tracemalloc_memory())

    print("Data Get :", new_data)
    print("Time it took to open data to add", end0-start0)
    print("Time it took to open storage", end1-start1)
    print("Time it took to concatenate", end2-end1)
    print("Time it took to get", end3-end2,)

    # tim_moy = 0
    # for i in range(10):
    #     start = time()
    #     print(new_data.uwnd[np.random.randint(192000), np.random.randint(100), np.random.randint(100), np.random.randint(30)].values)
    #     end = time()
    #     tim_moy += end-start
    # print('Time it took to get a value', tim_moy/10)

    start = time()
    print(new_data.uwnd[0, 0, 0, 0].values)
    end = time()
    print('Time it took to get first value', end-start)

    print('After get one value', tracemalloc.get_tracemalloc_memory())
    start = time()
    print(new_data.uwnd[100000, 0, 99, 15].values)
    end = time()
    print('Time it took to get middle value', end - start)

    start = time()
    print(new_data.uwnd[191999, 99, 99, 29].values)
    end = time()
    print('Time it took to get last value', end - start)
    print('After get three value', tracemalloc.get_tracemalloc_memory())
    tracemalloc.stop()

""" concat : il faut concatener dans la dimension où les coordonnées n'ont rien en commun, les autres se complètent ensuite"""

""" regarder coordonnées en commun
        si une coord rien en commun : concat là dessus (peut être à sort_by() sur la coordonnées)
            si toutes les coordonnées ont des valeurs en communs voir tout en commun"""

if __name__ == "__main__":
    request = create_prefilled_request()
    metadata = make_metadata_from_request_item(request)
    fetch(metadata=metadata)
    # got = get(metadata=metadata)
    # print(got)