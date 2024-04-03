import os
import sys
import multiprocessing as mp
import time

import wind_analysis

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wind_engine.minimalist_wind_engine import MinimalistWindEngine


def load_MinimalistWindEngine():
    DataLoader.WindEngine = MinimalistWindEngine()
    print("DataLoader : wind loaded")
    DataLoader.wa = wind_analysis.analysis_engine.analyse_wind(DataLoader.WindEngine.wind_data)
    print("DataLoader : wind analysed")
    DataLoader.wa_figures = wind_analysis.report_generator.get_analysis_figures(DataLoader.wa)
    print("DataLoader : analysis rendered")


class DataLoader:
    """Static class that aims to be shared by everyone.
    Can manage asychronous loading of data and say if yes or no data is loaded.
    TODO --> also pull the cities and countries data from here"""
    WindEngine = None
    wa = None
    wa_figures = None

    @staticmethod
    def load_and_analyse_wind():
        p = mp.Process(target=load_MinimalistWindEngine)
        p.start()

    @staticmethod
    def slow_load_and_analyse_wind():
        load_MinimalistWindEngine()

    @staticmethod
    def wind_has_not_loaded():
        return DataLoader.WindEngine is None


if __name__ == "__main__":
    t = time.perf_counter()
    DataLoader.load_and_analyse_wind()
    print(time.perf_counter() - t)
