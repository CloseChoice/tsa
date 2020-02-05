"""timeseries_utils provides general functionality for generating and processing time series"""
from functools import reduce
from typing import List, Union
import pandas as pd
import numpy as np

from sfc.modules.utils.enums import Interval


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
    """add two timeseries with different dates together, e.g. ts1 has daily indices, ts2 has monthly"""
    if len(ts1) > len(ts2):
        ts = ts1.add(ts2).combine_first(ts1)
    else:
        ts = ts2.add(ts1).combine_first(ts2)
    return ts

# USE FOLLOWING GENERATING FUNCTIONS


def ts_gaussian_seasonality(x, seasonality, noise=0.01, spread_factor=0.001):
    """ time series with a gaussian seasonality, which means that each seaonsality time window represents a gaussian
    density function.
    x = time range of the seasonality
    :param seasonality: seasonality used for sampling
    :param noise: noise added to the gaussian function """
    spread_factor = spread_factor * len(x)

    x = (x - np.min(x)) % seasonality
    x_spread = np.max(x) - np.min(x)
    l = 1 / (spread_factor * x_spread * np.sqrt(np.pi * 2))
    return (l * np.exp(
        -(x - x[int(x_spread / 2)]) ** 2 / (spread_factor * x_spread))) + noise * l * np.random.rand(len(x))


def create_ts_from_dist(start_dt: str, end_dt: str, time_interval: Interval, seasonality: int,
                        dist=ts_gaussian_seasonality) -> (np.array, np.array):
    """ sample a time series with seasonality from the function func. All native pandas timedeltas are supported."""
    x = pd.date_range(start_dt, end_dt, freq=f'1{time_interval.value}')
    x_ints = x.astype(np.int32) / (pd.Timedelta(f'1{time_interval.value}') / pd.Timedelta('1nanosecond'))
    y = dist(x_ints, seasonality)
    return pd.Series(y, index=x)


def agg_ts_from_dists(start_dt: str, end_dt: str, seasonalities: List[int], time_interval: Interval,
                      dist_list=None):
    """constructs a time series with given distributions and seasonalities in a given frequency time_interval"""
    ts_list = []
    dist_list = dist_list or [ts_gaussian_seasonality] * len(seasonalities)
    for dist, seasonality in zip(dist_list, seasonalities):
        ts_list.append(create_ts_from_dist(start_dt, end_dt, time_interval, seasonality, dist))
    ts = reduce(lambda x, y: add_ts_with_different_dates(x, y), ts_list)  # add time series together
    return ts


def peaks_by_rules(date_index: pd.DatetimeIndex, ts_value: Union[int, float], col_name: str, rule: str = 'day',
                   rule_value: Union[int, float, pd.DateOffset] = 1):
    """This function is capable of generating timeseries with non-zero values 'ts_value' if the rule is fulfilled
    and zero values at any other time.
    Allowed rules:
    - any of the attributes of pd.DateIndex
    - offsets
    - last_n_day_of_month

    Here some help in order you forget how to use the rules:
    - pd.DateOffset(month=1) will be just the january
    - more to come"""

    def last_n_day_of_month(time_stamp):
        return time_stamp == (time_stamp - pd.offsets.Day(1) + pd.offsets.MonthEnd()) - pd.offsets.Day(rule_value)

    if rule == 'last_n_day_of_month':
        assert 0 <= rule_value <= 31, f"rule_value out of range. Allowed values are between 0 and 31 but" \
                                      f" {rule_value} was given"
        values = np.where(pd.Series(date_index).apply(lambda x: last_n_day_of_month(x)), ts_value, 0)
    elif hasattr(date_index, rule):
        values = np.where(getattr(date_index, rule) == rule_value, ts_value, 0)
    elif isinstance(rule_value, pd.DateOffset) and rule == 'offsets':
        values = np.where(date_index + rule_value == date_index, ts_value, 0)
    else:
        raise NotImplementedError(f'your rule is not implemented. Value:{rule_value}, rule: {rule}')
    df = pd.DataFrame(values, columns=[col_name], index=date_index)
    return df


def split_categorical_time_series_labels(features: pd.DataFrame, labels: pd.Series, time_stamps: int,
                                         train_size: float):
    """Stratified split of time series with categorical labels."""
    def assertions():
        assert len(features) == len(labels)
        assert (features.index == np.arange(
            len(features))).all(), "Index must start at 0 and range up to len(df_features)-1"
        assert (labels.index == np.arange(len(labels))).all(), "Index must start at 0 and range up to len(labels)-1"
        assert 0 <= train_size <= 1, f"train size must be between 0 and 1 but is given as {train_size}"
        assert features.isnull().any(1).sum() == 0, "Null detected in features. Please impute before splitting"
        assert labels.isnull().all().sum() == 0, "Null detected in labels. Pleas impute before splitting"

    assertions()

    labels_train, labels_test = pd.Series(), pd.Series()
    features_train, features_test = pd.DataFrame(), pd.DataFrame()
    indices = [list(labels.loc[labels == i].index) for i in sorted(labels.unique())]
    for idxs in indices:
        features_label = reshape_array_to_tensor(features.loc[idxs], time_stamps)
        features_label = features_label.dropna()
        y_label = labels.loc[features_label.index]
        split_ind = int(len(y_label) * train_size)

        features_train = features_train.append(features_label.iloc[:split_ind])
        features_test = features_test.append(features_label.iloc[split_ind:])
        labels_train = labels_train.append(y_label.iloc[:split_ind])
        labels_test = labels_test.append(y_label.iloc[split_ind:])
    return features_train, features_test, labels_train, labels_test


def reshape_array_to_tensor(data: pd.DataFrame, time_steps: int):
    """reshape an array to tensor form. Shifts the dataframe for each element in 1-time_steps and appends the shifted
    dataframe on the right.
    :param data: given dataframe to reshape
    :param time_steps: time steps for which the data frame is shifted."""
    return pd.concat([data.shift(-i) for i in range(time_steps)], 1)
