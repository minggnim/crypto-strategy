# crypto-strategy
A repository to perform backtests and create trading strategies for cryptocurrencies.

[![Python package](https://github.com/minggnim/crypto-strategy/actions/workflows/python-package.yml/badge.svg)](https://github.com/minggnim/crypto-strategy/actions/workflows/python-package.yml)
[![pypi-upload](https://github.com/minggnim/crypto-strategy/actions/workflows/python-publish.yml/badge.svg)](https://github.com/minggnim/crypto-strategy/actions/workflows/python-publish.yml)
[![pages-build-deployment](https://github.com/minggnim/crypto-strategy/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/minggnim/crypto-strategy/actions/workflows/pages/pages-build-deployment)

![](./img/algo-trading.png)


## Install
```
pip install crypto-strategy[full]
```

## Usage
1. Moving average strategy
```
BestMaStrategy(symbols, freq, res_dir, flag_filter, flag_ts_stop)
```
- symbols: asset name, e.g., BTCUSDT
- freq: data frequency to use, 1h | 4h
- res_dir: results directory
- flag_filter: filter to use, [mmi | ang]
    - mmi: Market Meanness Index filter
    - ang: Linear Regression Angle filter
- flag_ts_stop: trailing stop filter


2. Breakout strategy
```
BestBoStrategy(symbols, freq, res_dir, flag_filter, flag_ts_stop)
```
- symbols: asset name, e.g., BTCUSDT
- freq: data frequency to use, 1h | 4h
- res_dir: results directory
- flag_filter: filter to use, [mmi | ang]
    - mmi: Market Meanness Index filter
    - ang: Linear Regression Angle filter
- flag_ts_stop: trailing stop filter

3. macd strategy
```
BestMacdStrategy(symbols, freq, res_dir, flag_filter)
```
- symbols: asset name, e.g., BTCUSDT
- freq: data frequency to use, 1h | 4h
- res_dir: results directory
- flag_filter: filter to use, [mmi | ang | stoch | sma]
    - vol: Volume filter
    - ang: Linear Regression Angle filter

## CLI

Backtests can also be carried out in command line. To find out more

```
crypto --help
```

## Tests
pytest