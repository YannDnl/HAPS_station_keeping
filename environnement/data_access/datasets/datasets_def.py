from environnement.data_access.datasets.noaa import *
from environnement.data_access.datasets.era5 import *

""" where to find the different functions associated to the different datasets """
datasets_def = {
                "NOAA": {
                    "API": fetch_wind_with_API_of_NOAA,
                    "one_day_dimensions": one_day_dimensions_of_NOAA,
                    "grid_memory_size": grid_memory_size_of_NOAA,
                    "storage_time_split": storage_time_split_NOAA,
                    "convert_time": convert_timestamp_in_datetime_NOAA,
                    "convert_datetime": convert_datetime_in_timestamp_NOAA,
                    "storage_file_total_size": storage_file_total_size_NOAA
                },
                "ERA5": {
                    "API": fetch_wind_with_API_of_ERA5,
                    "one_day_dimensions": one_day_dimensions_of_ERA5,
                    "grid_memory_size": grid_memory_size_of_ERA5,
                    "storage_time_split": storage_time_split_ERA5,
                    "convert_time": convert_timestamp_in_datetime_ERA5,
                    "convert_datetime": convert_datetime_in_timestamp_ERA5,
                    "storage_file_total_size": storage_file_total_size_ERA5
                }
}
            

"""

datsets_def = {
    sources{
        layout{
            variables{
                "nom de la variable"
                "description"
            }
        }
    }
}

il y a un module par source qui décrit comment tirer de la donnée de cette source (ERA5, NCEP-DOE-reanalysis-2, GFS, GFS-ens, GFS-archive)
Une source correspond à une manière de tirer de la donnée (via une API par exemple)

Une source peut voir ses données décrites par différents data_layouts. Ces derniers sont tous traqués dans un fichier dédié.

Chaque layout peut contenir plusieurs variables. Les variables ont un nom et une description. Elles sont a priori toutes en float16.

"""

