import os
import numpy as np
import pandas as pd
from crypto_strategy.data import download_crypto_history
from .base import (
    breakout_strategy,
    vol_filter, ang_filter,
    BestStrategy, InspectStrategy, CheckIndicators
)


RANGE_WINDOW = np.arange(10, 200, 5)
RANGE_TIMEPERIOD = np.arange(5, 55, 5)
RANGE_MULTIPLIER = np.arange(1, 10, 1)
RANGE_THRESHOLD = np.arange(0, 60, 5)
RANGE_TS_STOP = np.arange(0.1, 0.6, 0.05)


def create_vol_filter(flag_filter, **kwargs):
    filters = dict()
    if flag_filter:
        if kwargs:
            assert kwargs.get('timeperiod') and kwargs.get('multiplier'),\
                'Volume fitler doesn\'t have config params provided'
            filter_config = dict(
                timeperiod = kwargs['timeperiod'],
                multiplier = kwargs['multiplier']
            )
        else:
            filter_config = dict(
                timeperiod = RANGE_TIMEPERIOD,
                multiplier = RANGE_MULTIPLIER
            )
        filters = {
            'vol': vol_filter.create({
                'timeperiod': filter_config['timeperiod'],
                'multiplier': filter_config['multiplier']
            })
        }
    return filters


def create_ang_filter(flag_filter, **kwargs):
    filters = dict()
    if flag_filter:
        if kwargs:
            assert kwargs.get('timeperiod') and kwargs.get('threshold') is not None,\
                'Angle filter doesn\'t have config params provided'
            filter_config = dict(
                timeperiod = kwargs['timeperiod'],
                threshold = kwargs['threshold']
            )
        else:
            filter_config = dict(
                timeperiod = RANGE_TIMEPERIOD,
                threshold = RANGE_THRESHOLD
            )
        filters = {
            'ang': ang_filter.create({
                'timeperiod': filter_config['timeperiod'],
                'threshold': filter_config['threshold']
            })
        }
    return filters


def get_filter(flag_filter, **kwargs):
    filters = dict()
    if flag_filter == 'vol':
        filters = create_vol_filter(flag_filter, **kwargs)
    if flag_filter == 'ang':
        filters = create_ang_filter(flag_filter, **kwargs)
    return filters


def get_strategy(strategy):
    if strategy == 'bo':
        return breakout_strategy
    else:
        raise ValueError('Strategy not recognized. Please choose from bo and bo_rev')


def create_bo_variables(**kwargs):
    if kwargs:
        assert kwargs.get('long_window') and kwargs.get('short_window'), \
            'BO strategy doesn\'t have config params provided'
        variables = dict(
            long_window = kwargs['long_window'],
            short_window = kwargs['short_window']
            )
    else:
        variables = dict(
            long_window = RANGE_WINDOW,
            short_window = RANGE_WINDOW
        )
    return variables


def create_bo_variables_ts_stop(**kwargs):
    if kwargs:
        assert kwargs.get('long_window') and kwargs.get('short_window') and kwargs.get('ts_stop'), \
            'BO strategy doesn\'t have config params provided'
        variables = dict(
            long_window = kwargs['long_window'],
            short_window = kwargs['short_window'],
            ts_stop = kwargs['ts_stop']
        )
    else:
        variables = dict(
            long_window = RANGE_WINDOW,
            short_window = RANGE_WINDOW,
            ts_stop = RANGE_TS_STOP
        )
    return variables

class BestBoStrategy(BestStrategy):
    '''
    This class provides the method to optimize the BO strategy
    symbols: a list of symbols to be optimzied on, e.g., ['BTCUSDT']
    freq: currently supported values: '1h' or '4h'
    res_dir: the output directory
    flag_filter: currently supported fitlers: 'vol', 'ang', default: None
    flag_ts_stop: flag to turn on/off trailing stop
    strategy: 'bo'
    '''
    def __init__(self, symbols: list, freq: str, res_dir: str,
                 flag_filter: str = None,
                 flag_ts_stop: bool = False,
                 strategy: str = 'bo',
                 ):
        super().__init__(symbols, freq, res_dir, flag_filter, strategy)
        self.flag_ts_stop = flag_ts_stop
        self.generate_best_params()

    def _get_strategy(self, strategy):
        return get_strategy(strategy)

    def _get_filter(self, **kwargs):
        return get_filter(flag_filter=self.flag_filter, **kwargs)

    def _get_variables(self, **kwargs):
        if self.flag_ts_stop:
            return create_bo_variables_ts_stop(**kwargs)
        return create_bo_variables(**kwargs)

    def _get_grid_search(self):
        if self.flag_filter == 'vol':
            return self.grid_search_vol_params()
        elif self.flag_filter == 'ang':
            return self.grid_search_ang_params()
        else:
            return self.grid_search_params()

    def get_best_params(self, total_best_params, n=10):
        total_best_params = pd.DataFrame(total_best_params)
        total_best_params['sharpe'] = (total_best_params['sharpe'] + 0.05).round(1)
        total_best_params['gap'] = total_best_params['long_window'] - total_best_params['short_window']
        if 'advstex_ts_stop' in total_best_params.columns:
            total_best_params = (
                total_best_params
                .query('gap > 0')
                .sort_values(
                    by=['sharpe', 'gap', 'short_window', 'advstex_ts_stop'],
                    ascending=[True, True, False, False])
            )
        else:
            total_best_params = (
                total_best_params
                .query('gap > 0')
                .sort_values(
                    by=['sharpe', 'gap', 'short_window'],
                    ascending=[True, True, False]
                    )
            )

        print(total_best_params)
        return total_best_params.tail(1).to_dict(orient='records')[0] if not total_best_params.empty else None

    def grid_search_params(self):
        variables = self._get_variables()
        best_params = self.backtest(variables)
        return self.get_best_params(best_params)

    def grid_search_vol_params(self):
        variables = self._get_variables()
        total_best_params = list()
        for multiplier in RANGE_MULTIPLIER:
            for timeperiod in RANGE_TIMEPERIOD:
                filters = self._get_filter(timeperiod=timeperiod, multiplier=multiplier)
                best_params = self.backtest(variables, filters)
                total_best_params.extend(best_params)
        return self.get_best_params(total_best_params)

    def grid_search_ang_params(self):
        variables = self._get_variables()
        total_best_params = list()
        for threshold in RANGE_THRESHOLD:
            for timeperiod in RANGE_TIMEPERIOD:
                filters = self._get_filter(timeperiod=timeperiod, threshold=threshold)
                best_params = self.backtest(variables, filters)
                total_best_params.extend(best_params)
        return self.get_best_params(total_best_params)

    def apply_best_params(self, best_params, symbol):
        if self.flag_ts_stop:
            variables = self._get_variables(
                long_window=best_params['long_window'],
                short_window=best_params['short_window'],
                ts_stop=best_params['advstex_ts_stop']
            )
        else:
            variables = self._get_variables(
                long_window=best_params['long_window'],
                short_window=best_params['short_window']
            )
        filters = self._get_filter(
            timeperiod = best_params.get('vol_timeperiod') \
                if best_params.get('vol_timeperiod') \
                else best_params.get('ang_timeperiod'),
            multiplier = best_params.get('vol_multiplier'),
            threshold = best_params.get('ang_threshold')
            )
        portfolio = self.strategy.backtest(self.ohlcv, freq=self.freq, variables=variables, filters=filters)
        filename = f'''{symbol}-{self.freq}-{self.strategy_name}-{best_params['long_window']}-{best_params['short_window']}-'''
        if self.flag_ts_stop:
            filename += f'''ts-{best_params['advstex_ts_stop']:.2f}-'''
        if self.flag_filter == 'vol':
            filename += f'''{self.flag_filter}-{best_params['vol_timeperiod']}-{best_params['vol_multiplier']}-{self.date_str}.pkl'''
        elif self.flag_filter == 'ang':
            filename += f'''{self.flag_filter}-{best_params['ang_timeperiod']}-{best_params['ang_threshold']}-{self.date_str}.pkl'''
        else:
            filename += f'''{self.date_str}.pkl'''
        filename = os.path.join(self.output_path, filename)
        portfolio.stats().to_pickle(filename)

    def generate_best_params(self):
        print(f'The search for {self.symbols} starts now')
        for symbol in self.symbols:
            print(f'Start the search for {symbol}')
            self.ohlcv = download_crypto_history(symbol, self.freq)
            best_params = self._get_grid_search()
            if best_params:
                print(f'Found the best params {best_params}')
                self.apply_best_params(best_params, symbol)
            else:
                print('Not found the best params')
        print('The search for all the symbols is completed')


class CheckBoIndicators(CheckIndicators):
    '''
    This class provides the method to check Partial Differentiation
    and Combinatorially Symmetric Cross-validation of BO strategy.
    symbols: a list of symbols to be optimzied on, e.g., ['BTCUSDT']
    date: the date the best params are created
    res_dir: the output directory
    flag_filter: currently supported fitlers: 'vol', 'ang', default: None
    flag_ts_stop: flag to turn on/off trailing stop
    strategy: currently supported values: 'bo'
    '''
    def __init__(self,
                 symbols: list,
                 date: str,
                 res_dir: str,
                 flag_filter: str = None,
                 flag_ts_stop: bool = False,
                 strategy: str = 'bo'
                 ):
        super().__init__(symbols, date, res_dir, flag_filter, strategy)
        self.flag_ts_stop = flag_ts_stop
        self.check_indicators()

    def _get_strategy(self, strategy):
        return get_strategy(strategy)

    def _get_variables(self, **kwargs):
        if self.flag_ts_stop:
            return create_bo_variables_ts_stop(**kwargs)
        return create_bo_variables(**kwargs)

    def _get_filter(self, **kwargs):
        return get_filter(flag_filter=self.flag_filter, **kwargs)


class InspectBoStrategy(InspectStrategy):
    '''
    This class provides the method to optimize the BO strategy
    symbols: a list of symbols to be optimzied on, e.g., ['BTCUSDT']
    freq: currently supported values: '1h' or '4h'
    long_window, short_window: breakout params
    timeperiod, multiplier: volume filter params
    flag_filter: currently supported fitlers: 'vol', 'ang', default: None
    strategy: currently supports 'bo'
    '''
    def __init__(self, symbol: str, freq: str,
                 long_window: int, short_window: int, ts_stop: int = None,
                 timeperiod: int = None, multiplier: int = None, threshold: int = None,
                 flag_filter: str = None, flag_ts_stop: bool = False,
                 strategy: str = 'bo', show_fig: bool = True
                 ):
        super().__init__(symbol, freq, flag_filter, strategy, show_fig)
        self.long_window = long_window
        self.short_window = short_window
        self.multiplier = multiplier
        self.timeperiod = timeperiod
        self.threshold = threshold
        self.flag_ts_stop = flag_ts_stop
        self.ts_stop = ts_stop
        self.inspect()

    def _get_strategy(self, strategy):
        return get_strategy(strategy)

    def _get_variables(self, **kwargs):
        if self.flag_ts_stop:
            return create_bo_variables_ts_stop(
                long_window=self.long_window,
                short_window=self.short_window,
                ts_stop=self.ts_stop
            )
        return create_bo_variables(
            long_window=self.long_window,
            short_window=self.short_window
        )

    def _get_filter(self):
        return get_filter(
            self.flag_filter,
            timeperiod=self.timeperiod,
            multiplier=self.multiplier,
            threshold=self.threshold
            )


def returns_timeline(
    symbol, freq,
    long_window, short_window,
    strategy,
    ts_stop=None,
    timeperiod=None, multiplier=None, threshold=None, flag_filter=None,
    flag_ts_stop=False,
):
    ins = InspectBoStrategy(
        symbol=symbol,
        freq=freq,
        long_window=long_window,
        short_window=short_window,
        ts_stop=ts_stop,
        timeperiod=timeperiod,
        multiplier=multiplier,
        threshold=threshold,
        flag_filter=flag_filter,
        flag_ts_stop=flag_ts_stop,
        strategy=strategy,
        show_fig=False
    )
    daily_returns = ins.portfolio.daily_returns()
    acc_returns = {
        'Ret [:21-04-14]': (daily_returns[:'2021-04-15'] + 1).cumprod()[-1],
        'Ret [21-04-15:21-07-20]': (daily_returns['2021-04-15':'2021-07-21'] + 1).cumprod()[-1],
        'Ret [21-07-21:21-11-10]': (daily_returns['2021-07-21':'2021-11-11'] + 1).cumprod()[-1],
        'Ret [21-11-11:]': (daily_returns['2021-11-11':] + 1).cumprod()[-1]
    }
    return acc_returns
