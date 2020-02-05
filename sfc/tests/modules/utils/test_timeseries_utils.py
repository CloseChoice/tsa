import sfc.modules.utils.timeseries_to_dataframes as td
import sfc.modules.utils.timeseries_utils as tu
from sfc.modules.utils.enums import Interval

import numpy as np
from scipy.stats import norm, kappa4
from hypothesis import given, assume
import hypothesis.strategies as st
from hypothesis.strategies import integers
from hypothesis.extra.pandas import series, data_frames, columns, range_indexes
from hypothesis.extra.numpy import arrays
import pytest
import pandas as pd


def test_sample_from_distribution():
    gauss = norm()
    sampled = tu.sample_from_scipy_distribution(gauss, size=100)
    assert len(sampled) == 100


#def test_build_ts_by_seasonality():
#    gauss = norm()
#    ts = tu.build_ts_by_seasonality(gauss, '2019-01-01', '2019-04-01', 36, 'H')
#    assert ts.size == 2161
#    assert ts[ts != 0].size == 61
#
#def test_build_ts_from_distributions():
#    gauss = norm()
#    kappa = kappa4(0.1, 0.001)
#    dist_dict = {'gauss': gauss, 'kappa4': kappa}
#    ts = tu.build_ts_from_distributions(dist_dict, '2019-01-01', '2019-04-01', [36, 24], 'H')
#    assert ts[ts != 0].size == 121
#    assert ts[ts == 0].size == 2040

def test_agg_sample_n_gaussian_ts_as_df():
    df, seasonalities = td.sample_n_gaussian_ts_as_df(10, '2019-01-01', '2020-01-01', Interval.D)
    assert df.shape == (366, 10)

# globals for tests
index_len = 100
train_size_min = 0.1
train_size_max = 0.9


@given(features=data_frames(columns(['feat1', 'feat2', 'feat3'], dtype=float),
                            index=range_indexes(min_size=index_len, max_size=index_len)),
       labels=series(elements=st.integers(min_value=0, max_value=3),
                     index=range_indexes(min_size=index_len, max_size=index_len)),
       time_stamps=st.integers(min_value=1, max_value=10),
       train_size=st.floats(min_value=train_size_min, max_value=train_size_max, allow_infinity=False, allow_nan=False))
def test_split_categorical_time_series_labels(features: pd.DataFrame, labels: pd.Series, time_stamps: int,
                                              train_size: float):
    assume(train_size == round(train_size, 2))  # limit to only 2 digits after comma
    assume(features.isnull().any(1).sum() == 0)  # forbid nans
    features_train, features_test, labels_train, labels_test = \
        tu.split_categorical_time_series_labels(features, labels, time_stamps, train_size)
    assert index_len * train_size * 0.85 <= labels_train.size <= index_len * train_size * 1.15
    assert index_len * (1 - train_size) * 0.85 <= labels_test.size <= index_len * (1 - train_size) * 1.15
