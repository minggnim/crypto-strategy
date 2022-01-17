import numpy as np
from finlab_crypto.indicators import trends
from crypto_strategy.data import download_crypto_history, save_stats
from .base import (
    trend_strategy,
    mmi_filter, ang_filter,
    BestStrategy, InspectStrategy, CheckIndicators
)


RANGE_WINDOW = np.arange(30, 300, 10)
RANGE_TIMEPERIOD = np.arange(10, 50, 5)
RANGE_THRESHOLD = np.arange(0, 25, 5)


def create_mmi_filter(**kwargs):
    if kwargs:
        assert kwargs.get('timeperiod'),\
            'mmi fitler doesn\'t have config params provided'
        filter_config = kwargs
    else:
        filter_config = dict(
            timeperiod = RANGE_TIMEPERIOD,
        )
    filters = {
        'mmi': mmi_filter.create({
            'timeperiod': filter_config['timeperiod']
        })
    }
    return filters

def create_ang_filter(**kwargs):
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
    if flag_filter == 'mmi':
        filters = create_mmi_filter(**kwargs)
    if flag_filter == 'ang':
        filters = create_ang_filter(**kwargs)
    return filters

def create_variables(**kwargs):
    name = kwargs.get('name')
    if name:
        variables = dict(name = name)
    else:
        raise ValueError('The name of the MA strategy is required')
    if kwargs.get('n1') and kwargs.get('n2'):
        variables.update(kwargs)
    elif kwargs.get('n1') or kwargs.get('n2'):
        raise ValueError('Only one of n1 and n2 is provided')
    else:
        variables['n1'] = RANGE_WINDOW
        variables['n2'] = RANGE_WINDOW
    return variables


class BestMaStrategy(BestStrategy):
    '''
    This class provides the method to optimize the MA strategy
    symbols: a list of symbols to be optimzied on, e.g., ['BTCUSDT']
    freq: currently supported values are '1h' or '4h'
    res_dir: the output directory
    flag_fitler: currently supported fitlers: 'mmi', 'ang', default: None
    trends: a list of MA strategies, default: trends.keys()
    strategy: strategy name, default: 'ma'
    '''
    def __init__(self, symbols: list, freq: str, res_dir: str,
                 flag_filter: str = None,
                 trends: list = trends.keys(),
                 strategy: str = 'ma'
                 ):
        super().__init__(symbols, freq, res_dir, flag_filter, strategy)
        self.trends = trends
        self.generate_best_params()

    def _get_strategy(self, strategy):
        return trend_strategy

    def _get_filter(self, **kwargs):
        return get_filter(flag_filter=self.flag_filter, **kwargs)

    def _get_variables(self, **kwargs):
        return create_variables(**kwargs)

    def _get_grid_search(self):
        if self.flag_filter == 'mmi':
            return self.grid_search_mmi_params()
        if self.flag_filter == 'ang':
            return self.grid_search_ang_params()
        return self.grid_search_params()

    def grid_search_params(self):
        variables = self._get_variables(name=self.trends)
        best_params = self.backtest(variables)
        return self.get_best_params(best_params)

    def grid_search_mmi_params(self):
        variables = self._get_variables(name=self.trends)
        total_best_params = list()
        for timeperiod in RANGE_TIMEPERIOD:
            filters = self._get_filter(timeperiod=timeperiod)
            best_params = self.backtest(variables, filters)
            total_best_params.extend(best_params)
        return self.get_best_params(total_best_params)

    def grid_search_ang_params(self):
        variables = self._get_variables(name=self.trends)
        total_best_params = list()
        for threshold in RANGE_THRESHOLD:
            for timeperiod in RANGE_TIMEPERIOD:
                filters = self._get_filter(timeperiod=timeperiod, threshold=threshold)
                best_params = self.backtest(variables, filters)
                total_best_params.extend(best_params)
        return self.get_best_params(total_best_params)

    def apply_best_params(self, best_params, symbol):
        variables = self._get_variables(name=best_params['name'], n1=best_params['n1'], n2=best_params['n2'])
        filters = self._get_filter(
            timeperiod = best_params.get('mmi_timeperiod') \
                or best_params.get('ang_timeperiod'),
            threshold = best_params.get('ang_threshold')
            )
        portfolio = self.strategy.backtest(self.ohlcv, variables=variables, filters=filters, freq=self.freq)
        filename = f"{symbol}-{self.freq}-{best_params['name']}-{best_params['n1']}-{best_params['n2']}-"
        if self.flag_filter == 'mmi':
            filename += f"{self.flag_filter}-{best_params['mmi_timeperiod']}-{self.date_str}.pkl"
        elif self.flag_filter == 'ang':
            filename += f"{self.flag_filter}-{best_params['ang_timeperiod']}-{best_params['ang_threshold']}-{self.date_str}.pkl"
        else:
            filename += f"{self.date_str}.pkl"
        save_stats(portfolio.stats(), self.output_path, filename)
        print(f'The stats are saved to {self.output_path}/{filename}')

    def generate_best_params(self):
        print(f'The search for {self.symbols} starts now')
        for symbol in self.symbols:
            print(f'Start the search for {symbol}')
            self.ohlcv = download_crypto_history(symbol, self.freq)
            best_params = self._get_grid_search()
            print(f'Found the best params {best_params}')
            self.apply_best_params(best_params, symbol)
        print('The search for all the symbols is completed')


class CheckMaIndicators(CheckIndicators):
    '''
    This class provides the method to check Partial Differentiation
    and Combinatorially Symmetric Cross-validation of MA strategy.
    symbols: a list of symbols to be optimzied on, e.g., ['BTCUSDT']
    date: the date the best params are created
    res_dir: the output directory
    name: the name of the MA strategy
    flag_fitler: currently supported fitlers: 'mmi', 'ang', default: None
    '''
    def __init__(self, symbols: list, date: str, res_dir: str,
                 flag_filter: str = None, strategy: str = 'ma'
                 ):
        super().__init__(symbols, date, res_dir, flag_filter, strategy)
        self.check_indicators()

    def _get_strategy(self, strategy):
        return trend_strategy

    def _get_variables(self, **kwargs):
        return create_variables(**kwargs)

    def _get_filter(self, **kwargs):
        return get_filter(flag_filter=self.flag_filter, **kwargs)


class InspectMaStrategy(InspectStrategy):
    '''
    This class provides a method to inspect the MA strategy with given params
    symbol: the name of the crypto, e.g., 'BTCUSDT'
    freq: currently supported values are '1h' or '4h'
    name, n1, n2: the name and the params of the ma strategy, e.g., 'sma', 100, 50
    timeperiod: param used in either mmi or ang filter
    threshold: param used in ang filter
    flag_fitler: currently supported fitlers: 'mmi', 'ang', default: None
    '''
    def __init__(self,
                symbol: str, freq: str,
                name: str, n1: int, n2: int,
                timeperiod: int = None,
                threshold: int = None,
                flag_filter: str = None,
                strategy: str = 'ma',
                show_fig: bool = True
                ):
        super().__init__(symbol, freq, flag_filter, strategy, show_fig)
        self.name = name
        self.n1 = n1
        self.n2 = n2
        self.timeperiod = timeperiod
        self.threshold = threshold
        self.flag_filter = flag_filter
        self.inspect()

    def _get_strategy(self, strategy):
        return trend_strategy

    def _get_variables(self):
        return create_variables(
            name=self.name,
            n1=self.n1,
            n2=self.n2
            )

    def _get_filter(self):
        return get_filter(
            self.flag_filter,
            timeperiod=self.timeperiod,
            threshold=self.threshold
            )
