import pytest
import numpy as np
import pandas as pd
import vectorbt as vbt
from finlab_crypto.utility import stop_early


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

def test_ts_stop(price, entries, exits, stop_vars):
    entries_after, exits_after = stop_early(price, entries, exits, stop_vars)
    assert (entries.values == entries_after.values).all()
    assert (exits_after.values.squeeze() == [False, False, False, True, False, False]).all()