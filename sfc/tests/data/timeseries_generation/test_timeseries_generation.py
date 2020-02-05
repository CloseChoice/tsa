#import utils.timeseries_utils as tu
from scipy.stats import norm, kappa4
import pytest


@pytest.mark.skip(reason="check whether tests are still needed")
def test_sample_from_distribution():
    gauss = norm()
    sampled = tu.sample_from_scipy_distribution(gauss, size=100)
    assert len(sampled) == 100


@pytest.mark.skip(reason="check whether tests are still needed")
def test_build_ts_by_seasonality():
    gauss = norm()
    ts = tu.build_ts_by_seasonality(gauss, '2019-01-01', '2019-04-01', 36, 'H')
    assert ts.size == 2161
    assert ts[ts != 0].size == 61

@pytest.mark.skip(reason="check whether tests are still needed")
def test_build_ts_from_distributions():
    gauss = norm()
    kappa = kappa4(0.1, 0.001)
    dist_dict = {'gauss': gauss, 'kappa4': kappa}
    ts = tu.build_ts_from_distributions(dist_dict, '2019-01-01', '2019-04-01', [36, 24], 'H')
    assert ts[ts != 0].size == 121
    assert ts[ts == 0].size == 2040
