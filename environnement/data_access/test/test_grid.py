import unittest
from datetime import datetime, timedelta
import numpy as np
import calendar
from environnement.data_access.request_build.grid import grid_date
from utils import convert_datetime_in_timestamp

class TestGridDate(unittest.TestCase):
    def test_grid_date(self):
        # Test case 1: Test with month_sbs = [1], day_sbs = [1], hour_sbs = [1]
        bounds = [datetime(2023, 1, 1), datetime(2023, 1, 2)]
        month_sbs = [1]
        day_sbs = [1]
        hour_sbs = [1]
        one_day_dimensions = {'time': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])}
        expected_output = [convert_datetime_in_timestamp(datetime(2023, 1, 1, hour=0)),
                           convert_datetime_in_timestamp(datetime(2023, 1, 1, hour=1)),
                           convert_datetime_in_timestamp(datetime(2023, 1, 1, hour=2)),
                           convert_datetime_in_timestamp(datetime(2023, 1, 1, hour=3)),
                           convert_datetime_in_timestamp(datetime(2023, 1, 1, hour=4)),
                           convert_datetime_in_timestamp(datetime(2023, 1, 1, hour=5)),
                           convert_datetime_in_timestamp(datetime(2023, 1, 1, hour=6)),
                           convert_datetime_in_timestamp(datetime(2023, 1, 1, hour=7)),
                           convert_datetime_in_timestamp(datetime(2023, 1, 1, hour=8)),
                           convert_datetime_in_timestamp(datetime(2023, 1, 1, hour=9)),
                           convert_datetime_in_timestamp(datetime(2023, 1, 1, hour=10)),
                           convert_datetime_in_timestamp(datetime(2023, 1, 2, hour=0))]
        output = grid_date(bounds, month_sbs, day_sbs, hour_sbs, one_day_dimensions)
        self.assertListEqual(output, expected_output)

        # Test case 2: Test with different subsampling values
        bounds = [datetime(2023, 1, 1), datetime(2023, 1, 3)]
        month_sbs = [2]
        day_sbs = [2]
        hour_sbs = [3]
        one_day_dimensions = {'time': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])}
        expected_output = [convert_datetime_in_timestamp(datetime(2023, 1, 1, hour=0)),
                           convert_datetime_in_timestamp(datetime(2023, 1, 1, hour=3)),
                           convert_datetime_in_timestamp(datetime(2023, 1, 1, hour=6)),
                           convert_datetime_in_timestamp(datetime(2023, 1, 1, hour=9)),
                           convert_datetime_in_timestamp(datetime(2023,1,3,0))]
        output = grid_date(bounds, month_sbs, day_sbs, hour_sbs, one_day_dimensions)
        self.assertListEqual(output, expected_output)

if __name__ == '__main__':
    unittest.main()