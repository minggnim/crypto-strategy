import pytest
import pandas as pd
import numpy as np
from crypto_strategy.strategies.bo_strategy import InspectBoStrategy


@pytest.fixture
def positions_prior():
    return pd.read_pickle('./tests/positions_XRPUSDT-4h-bo-100-15-ts_stop-0.10-vol-10-2-20211230.pkl')


def test_regression(positions_prior):
    stats = InspectBoStrategy(
        'XRPUSDT', '4h', 
        long_window=100, short_window=15, 
        flag_ts_stop=True, ts_stop=0.1, 
        flag_filter='vol', timeperiod=10, multiplier=2, show_fig=False)

    positions = stats.portfolio.positions.records_readable
    positions = positions[positions["Entry Timestamp"] < "2021-12-30"]
    np.testing.assert_allclose(positions["Size"].values, positions_prior["Size"].values)
    np.testing.assert_equal(positions['Entry Timestamp'].values, positions_prior['Entry Date'].values)