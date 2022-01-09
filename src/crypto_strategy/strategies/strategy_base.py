import os
import pathlib
from datetime import datetime
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from crypto_strategy.plot import plot_indicators
from crypto_strategy.data import check_and_create_dir, download_crypto_history


class BestStrategy(ABC):
    def __init__(self, symbols: list, freq: list, res_dir: str, flag_filter: str, strategy: str):
        self._sanity_check(flag_filter, res_dir)
        self.symbols = symbols
        self.freq = freq
        self.flag_filter = flag_filter
        self.strategy_name = strategy
        self.strategy = self._get_strategy(strategy)
        self.date_str = datetime.today().date().strftime("%Y-%m-%d")
        self.output_path = check_and_create_dir(res_dir, self.date_str.replace('-',''))
        self.output_path = pathlib.Path(self.output_path)

    @staticmethod
    def _sanity_check(flag_filter, res_dir):
        if flag_filter and flag_filter not in res_dir:
            raise ValueError(f'The res_dir name should contain {flag_filter} filter')
        if not flag_filter and 'filter' in res_dir:
            raise ValueError(f'The res_dir name should not contain filter name')
    
    @abstractmethod
    def _get_strategy(self, strategy):
        pass

    @abstractmethod
    def grid_search_params(self):
        pass

    @abstractmethod 
    def _get_grid_search(self):
        pass

    @abstractmethod
    def apply_best_params(self):
        pass

    def get_best_params(self, total_best_params, n=10):
        total_best_params = (
            pd.DataFrame(total_best_params)
            .sort_values(by='sharpe')
        )
        print(total_best_params.head(n))
        return total_best_params.tail(1).to_dict(orient='records')[0]

    def backtest(self, variables, filters=dict(), n=10):
        portfolio = self.strategy.backtest(self.ohlcv, freq=self.freq, variables=variables, filters=filters)
        best_params = portfolio.sharpe_ratio().replace([np.inf, -np.inf], np.nan).dropna().nlargest(n)
        best_params = (
            best_params
            .rename('sharpe')
            .reset_index()
            .to_dict(orient='records')
        )
        return best_params
    
    def generate_best_params(self):
        print(f'The search for {self.symbols} starts now')
        for symbol in self.symbols:
            print(f'Start the search for {symbol}')
            self.ohlcv = download_crypto_history(symbol, self.freq)
            best_params = self._get_grid_search()
            print(f'Found the best params {best_params}')
            self.apply_best_params(best_params, symbol)
        print('The search for all the symbols is completed')


class CheckIndicators(ABC):
    def __init__(self, 
                symbols: list, 
                date: str, 
                res_dir: str, 
                flag_filter: str, 
                strategy: str, 
                show_fig: bool = False):
        self.symbols = symbols
        self.date = date
        self.flag_filter = flag_filter
        self.res_dir = os.path.join(res_dir, date.replace('-', ''))
        self.strategy = self._get_strategy(strategy)
        self.show_fig = show_fig

    @abstractmethod
    def _get_strategy(self, strategy):
        pass
    
    @abstractmethod
    def _get_variables(self):
        pass

    @abstractmethod
    def _get_filter(self):
        pass

    def check_indicators(self):
        print(f'Start checking the optimized params for {self.symbols}')
        for symbol in self.symbols:
            if os.path.isdir(self.res_dir) and symbol:
                res = os.listdir(self.res_dir)
                for r in res:
                    if r.startswith(symbol) and r.endswith(self.date+'.pkl'):
                        comp = r.split('-')
                        assert symbol == comp[0], 'asset name doesn\'t match the file name'
                        freq = comp[1]
                        self.trend = comp[2] if comp[2].isalpha() else None
                        ohlcv = download_crypto_history(symbol, freq)
                        if self.trend:
                            variables = self._get_variables(name=self.trend)
                        else:
                            variables = self._get_variables()
                        filters = self._get_filter()
                        portfolio = self.strategy.backtest(ohlcv, freq=freq, variables=variables, filters=filters)
                        filename = f'{symbol}-{freq}-{self.trend}-{self.date}'
                        plot_indicators(portfolio, self.res_dir, filename, self.show_fig)


class InspectStrategy(ABC):
    def __init__(self, 
                symbol: str, 
                freq: str, 
                flag_filter: str, 
                strategy: str, 
                show_fig: bool = True):
        self.symbol = symbol
        self.freq = freq
        self.flag_filter = flag_filter
        self.strategy = self._get_strategy(strategy)
        self.show_fig = show_fig
        self.ohlcv = download_crypto_history(symbol, freq)

    @abstractmethod
    def _get_strategy(self, strategy):
        pass

    @abstractmethod
    def _get_variables(self):
        pass

    @abstractmethod
    def _get_filter(self):
        pass 

    def inspect(self):
        self.portfolio = self.strategy.backtest(
            self.ohlcv, 
            variables=self._get_variables(),
            filters=self._get_filter(),
            freq=self.freq,
            plot=self.show_fig)
