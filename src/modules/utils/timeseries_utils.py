import pandas as pd
import numpy as np
from functools import reduce


def sample_from_scipy_distribution(dist, size, **kwargs):
    """ use a given distribution to and extract size samples from it."""
    if kwargs:
       return dist.rvs(**kwargs, size=size)
    return dist.rvs(size=size)


def build_sparse_ts_by_seasonality(dist, start_date, end_date, seasonality, time_interval, **kwargs):
    """ builds a single seasonality time series between start and end_date. The time series is sampled from the
    given distribution dist"""

    def calculate_periods():
        return int((pd.to_datetime(end_date) - pd.to_datetime(start_date)) /
                   pd.to_timedelta(f'1{time_interval.lower()}')/seasonality)

    num_periods = calculate_periods()

    ts_ind = pd.date_range(start_date, end_date, periods=num_periods+1)
    values = sample_from_scipy_distribution(dist, size=ts_ind.size, **kwargs)
    ts = pd.Series(values, index=ts_ind).resample(time_interval).sum()
    return ts


def build_sparse_ts_from_distributions(start_date, end_date, seasonalities, time_interval, dist_dict, **kwargs):
    """constructs a time series with given distributions and seasonalities in a given frequency time_interval"""
    ts_list = []
    for (name, dist), seasonality in zip(dist_dict.items(), seasonalities):
        ts_list.append(build_sparse_ts_by_seasonality(dist, start_date, end_date, seasonality, time_interval,
                                            **kwargs.get(name, {})))
    ts = reduce(lambda x, y: add_ts_with_different_dates(x, y), ts_list)  # add time series together
    return ts


def build_dense_ts_by_distribution(start_date, end_date, seasonality, freq):
    """ Deprecated """
    dates = pd.date_range(start_date, end_date, freq=freq)
    omega = 2*np.pi/seasonality
    x = np.arange(len(dates))
    values = -1*np.cos(omega*x)+1
    # add random noise based on the values
    noise = np.random.rand(len(values))
    values += 0.1*noise*values
    return pd.Series(values, index=dates)


def add_ts_with_different_dates(ts1, ts2):
    if len(ts1) > len(ts2):
        ts = ts1.add(ts2).combine_first(ts1)
    else:
        ts = ts2.add(ts1).combine_first(ts2)
    return ts


def build_ts_id_with_firsts_units(dist, start_date, end_date, unit):
    """construct a new time series with entries at the first of the given unit, e.g. first of the month"""

    #def calculate_units_in_time_range():

    #time_idx = [start_date]


def build_ts_from_gaussian_ts(periods, variances):
    """build dense time series where points are distributed over a period as a gaussian curve. This should
    be extended in the future, so that arbitrary distributions can be sampled but this shall be enough as of
    November 2019."""

def ts_gaussian_seasonality(x, seasonality, noise=0.01, spread_factor=0.001):
    """ time series with a gaussian seasonality, which means that each seaonsality time window represents a gaussian
    density function.
    x = time range of the seasonality
    :param seasonality: seasonality used for sampling
    :param noise: noise added to the gaussian function """
    spread_factor = spread_factor * len(x)

    x = (x - np.min(x)) % seasonality
    x_spread = np.max(x) - np.min(x)
    return (l := 1 / (spread_factor * x_spread * np.sqrt(np.pi * 2)) * np.exp(
        -(x - x[int(x_spread / 2)]) ** 2 / (spread_factor * x_spread))) + noise * l * np.random.rand(len(x))


def create_ts_from_dist(start_dt, end_dt, time_interval, seasonality, dist=ts_gaussian_seasonality):
    """sample a time series with seasonality from the function func. All native pandas timedeltas are supported."""
    x = pd.date_range(start_dt, end_dt, freq=f'1{time_interval.lower()}')
    x_ints = x.astype(np.int32) / (pd.Timedelta(f'1{time_interval.lower()}') / pd.Timedelta('1nanosecond'))
    y = dist(x_ints, seasonality)
    return x, y

#def build_features_for_time_series(ts, seasonalities)

