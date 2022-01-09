import talib
from numbers import Number
from finlab_crypto.strategy import Strategy
from finlab_crypto.indicators import trends


@Strategy(name='sma', n1=20, n2=40)
def trend_strategy(ohlcv):
    name = trend_strategy.name
    n1 = trend_strategy.n1
    n2 = trend_strategy.n2
    filtered1 = trends[name](ohlcv.close, n1)
    filtered2 = trends[name](ohlcv.close, n2)
    entries = (filtered1 > filtered2) & (filtered1.shift() < filtered2.shift())
    exit = (filtered1 < filtered2) & (filtered1.shift() > filtered2.shift()) 
    figures = {
        'overlaps': {
            'trend1': filtered1,
            'trend2': filtered2,
        }
    }
    return entries, exit, figures

@Strategy(long_window=30, short_window=30)
def breakout_strategy(ohlcv):
    lw = breakout_strategy.long_window
    sw = breakout_strategy.short_window    
    ub = ohlcv.close.rolling(lw).max()
    lb = ohlcv.close.rolling(sw).min()
    entries = ohlcv.close == ub
    exits = ohlcv.close == lb
    figures = {
        'overlaps': {
            'ub': ub,
            'lb': lb
        }
    }
    return entries, exits, figures

@Strategy(long_window=30, short_window=30)
def breakout_strategy_revised(ohlcv):
    lw = breakout_strategy_revised.long_window
    sw = breakout_strategy_revised.short_window    
    ub = ohlcv.high.rolling(lw).max()
    lb = ohlcv.low.rolling(sw).min()
    # Break through new high to buy
    entries = ohlcv.high >= ub
    # Break through new low to sell
    exits = ohlcv.low <= lb
    figures = {
        'overlaps': {
            'ub': ub,
            'lb': lb
        }
    }
    return entries, exits, figures


# macd strategy
@Strategy(fastperiod=12, slowperiod=26, signalperiod=9)
def macd_strategy(ohlcv):
    macd, signal, macdhist = talib.MACD(
        ohlcv.close, 
        fastperiod=macd_strategy.fastperiod, 
        slowperiod=macd_strategy.slowperiod, 
        signalperiod=macd_strategy.signalperiod
    )

    entries = (macdhist > 0) & (macdhist.shift() < 0)
    exits = (macdhist < 0) & (macdhist.shift() > 0)
    figures = {
        'figures':{
            'macdhist': macdhist
        }
    }
    return entries, exits, figures

@Strategy(fastperiod=12, slowperiod=26, signalperiod=9)
def macd_strategy_revised(ohlcv):
    macd, signal, macdhist = talib.MACD(
        ohlcv.close, 
        fastperiod=macd_strategy.fastperiod, 
        slowperiod=macd_strategy.slowperiod, 
        signalperiod=macd_strategy.signalperiod
    )
    entries = (macdhist > 0) & (macdhist.shift() < 0) & (macd > 0)
    exits = (macdhist < 0) & (macdhist.shift() > 0) & (macd < 0)
    figures = {
        'macd':{
            'macdhist': macdhist
            }
        }
    return entries, exits, figures

@Strategy(timeperiod=14, buy_threshold=52, sell_threshold=50)
def rsi_strategy(ohlcv):
    rsi = talib.RSI(ohlcv.close, timeperiod=rsi_strategy.timeperiod)
    entries = (rsi > rsi_strategy.buy_threshold) & (rsi.shift() < rsi_strategy.buy_threshold)
    exits = (rsi < rsi_strategy.sell_threshold) & (rsi.shift() > rsi_strategy.sell_threshold)
    figure = {
        'figures': {
            'rsi': rsi
        }
    }
    return entries, exits, figure