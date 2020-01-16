""" Sample large amounts of time series into df so that ML algorithms can be applied on it."""
import pandas as pd
import numpy as np
from sfc.modules.utils.Enums import Interval

from sfc.modules.utils.timeseries_utils import agg_ts_from_dists


def sample_n_gaussian_ts_as_df(n: int, start_dt: str, end_dt: str, time_interval: Interval, dist_list=None,
                               seasonality_list=None):
   """:param """
   max_seasonality = int((pd.Timestamp(end_dt) - pd.Timestamp(start_dt))/(pd.Timedelta(f'1{time_interval.value}')))
   num_seas = np.random.choice(3, n) + 1
   df = pd.DataFrame()

   seasonality_list = seasonality_list or range(10, int(max_seasonality/10))
   seasonalities = [[j for j in np.random.choice(seasonality_list, k, replace=False)] for k in num_seas]
   for i in range(n):
      seas = seasonalities[i]
      df = pd.concat([df, agg_ts_from_dists(start_dt, end_dt, seas, time_interval, dist_list)], 1)
   df.columns = list(range(n))
   return df, seasonalities




