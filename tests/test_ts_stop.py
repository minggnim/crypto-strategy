import pytest
import numpy as np
import pandas as pd
import vectorbt as vbt
from finlab_crypto.utility import (
    stop_early,
    enumerate_variables,
    migrate_trailing_stop
)


@pytest.fixture
def price():
    price = pd.DataFrame({
        'open': [10, 11, 12, 11, 10, 9],
        'high': [11, 12, 13, 12, 11, 10],
        'low': [9, 10, 11, 10, 10, 8],
        'close': [10, 11, 12, 11, 10, 9]
    })
    return price


@pytest.fixture
def entries():
    return pd.DataFrame([True] + [False]*5)


@pytest.fixture
def exits():
    return pd.DataFrame([False] * 6)


@pytest.fixture
def stop_vars():
    return {'ts_stop': 0.1}


@pytest.fixture
def transform_stop_vars(stop_vars):
    stop_vars = enumerate_variables(stop_vars)
    stop_vars = {key: [stop_vars[i][key] for i in range(len(stop_vars))] for key in stop_vars[0].keys()}
    stop_vars = migrate_trailing_stop(stop_vars)
    return stop_vars


def test_transform_stop_vars(transform_stop_vars):
    assert transform_stop_vars == {'sl_stop': [0.1], 'sl_trail': [True], 'ts_stop': [0.1]}


def test_ohlcstx(entries, price, transform_stop_vars):
    ohlcstx = vbt.OHLCSTX.run(
        entries,
        price['open'],
        price['high'],
        price['low'],
        price['close'],
        **transform_stop_vars,
    )
    np.testing.assert_array_equal(
        ohlcstx.exits.squeeze(),
        [False, False, False, True, False, False]
        )
    np.testing.assert_allclose(
        ohlcstx.stop_price.squeeze(),
        [np.nan, np.nan, np.nan, 11.7, np.nan, np.nan],
        rtol=1e-10, atol=0
        )
    np.testing.assert_array_equal(
        ohlcstx.stop_type_readable.squeeze(),
        [None, None, None, 'TrailStop', None, None]
        )


def test_ts_stop(price, entries, exits, stop_vars):
    entries_after, exits_after = stop_early(price, entries, exits, stop_vars)
    assert (entries.values == entries_after.values).all()
    assert (exits_after.values.squeeze() == [False, False, False, True, False, False]).all()
