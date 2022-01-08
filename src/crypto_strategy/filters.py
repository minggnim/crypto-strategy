import talib
from finlab_crypto.strategy import Filter


@Filter(timeperiod=20)
def mmi_filter(ohlcv):
    median = ohlcv.close.rolling(mmi_filter.timeperiod).median()
    p1 = ohlcv.close > median
    p2 = ohlcv.close.shift() > median
    mmi = (p1 & p2).astype(int).rolling(mmi_filter.timeperiod).mean()
    figures = {
      'figures': {
          'mmi_index': mmi
      }
    }
    return mmi > 0.5, figures

@Filter(timeperiod=20)
def mmi_filter_ge_half(ohlcv):
    median = ohlcv.close.rolling(mmi_filter_ge_half.timeperiod).median()
    p1 = ohlcv.close > median
    p2 = ohlcv.close.shift() > median
    mmi = (p1 & p2).astype(int).rolling(mmi_filter_ge_half.timeperiod).mean()
    figures = {
      'figures': {
          'mmi_ge0.5_index': mmi
      }
    }
    return mmi >= 0.5, figures

@Filter(timeperiod=20, multiplier=2)
def vol_filter(ohlcv):
    vol_mean = ohlcv.volume.rolling(vol_filter.timeperiod).mean()
    vol = ohlcv.volume / (vol_filter.multiplier * vol_mean)
    figures = {
      'figures': {
          'vol_index': vol
      }
    }
    return vol >= 1, figures

@Filter(timeperiod=20, threshold=0)
def ang_filter(ohlcv):
    ang = talib.LINEARREG_ANGLE(ohlcv.close, ang_filter.timeperiod)
    figures = {
        'figures': {
            'ang_index': ang
        }
    }
    return ang > ang_filter.threshold, figures


@Filter(side='long', fast=5, slow=3, matype=0)
def stoch_filter(ohlcv):
    side = stoch_filter.side
    fast = stoch_filter.fast
    slow = stoch_filter.slow
    matype = stoch_filter.matype
    k, d = talib.STOCH(ohlcv.high, ohlcv.low, ohlcv.close,
                fastk_period=fast, slowk_period=slow, slowk_matype=matype, 
                slowd_period=slow, slowd_matype=matype)
    signals = (k > d) & (k > 50) & (k < 50) if side == 'long' else k < d
    fig = {
        'figures': {
            'kd': {'k': k, 'd': d}
        }
    }
    return signals, fig


@Filter(timeperiod=50)
def sma_filter(ohlcv):
    sma = talib.SMA(ohlcv.close, timeperiod=sma_filter.timeperiod)
    figures = {
      'figures': {
          'sma': sma
      }
    }
    return ohlcv.close > sma, figures