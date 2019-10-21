import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset


def sample_from_scipy_distribution(dist, size, **kwargs):
    """ use a given distribution to and extract size samples from it."""
    if kwargs:
       return dist.rvs(**kwargs, size=size)
    return dist.rvs(size=size)


def build_sparse_ts_by_seasonality(dist, start_date, end_date, seasonality, freq, **kwargs):
    """ builds a single seasonality time series between start and end_date. The time series is sampled from the
    given distribution dist"""

    def calculate_periods():
        return int((pd.to_datetime(end_date) - pd.to_datetime(start_date)) /
                   pd.to_timedelta(f'1{freq.lower()}')/seasonality)

    num_periods = calculate_periods()

    ts_ind = pd.date_range(start_date, end_date, periods=num_periods+1)
    values = sample_from_scipy_distribution(dist, size=ts_ind.size, **kwargs)
    ts = pd.Series(values, index=ts_ind).resample(freq).sum()
    return ts


def build_sparse_ts_from_distributions(dist_dict, start_date, end_date, seasonalities, freq, **kwargs):
    """constructs a time series with given distributions and seasonalities in a given frequency freq"""
    ts = pd.Series()
    for (name, dist), seasonality in zip(dist_dict.items(), seasonalities):
        if kwargs.get(name):
            ts = build_sparse_ts_by_seasonality(dist, start_date, end_date, seasonality, freq, **kwargs[name])\
                .add(ts, fill_value=0)
        else:
            ts = build_sparse_ts_by_seasonality(dist, start_date, end_date, seasonality, freq).add(ts, fill_value=0)
    return ts


def build_dense_ts_by_distribution(start_date, end_date, seasonality, freq):
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

    def calculate_number_of_units_in_time_range():
        _start_date = pd.to_datetime(start_date)
        _end_date = pd.to_datetime(end_date)
        max_range = (_end_date - _start_date)/pd.to_timedelta(f'28{unit}')
        return len([_start_date + DateOffset(months=i) for i in range(max_range) if
                    _start_date + DateOffset(months=i) < _end_date])

    time_idx = [start_date]


#def build_features_for_time_series(ts, seasonalities)
