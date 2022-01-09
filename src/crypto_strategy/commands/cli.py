import click
from crypto_strategy.ma_strategy import BestMaStrategy, CheckMaIndicators
from crypto_strategy.bo_strategy import BestBoStrategy, CheckBoIndicators
from crypto_strategy.reporting import generate_report as _generate_report


SYMBOLS = [
    'BTC', 'ETH', 'BNB', 'SOL', 'ADA',
    'XRP', 'DOT', 'DOGE', 'AVAX', 'LUNA',
    'LTC', 'UNI', 'LINK', 'ALGO', 'MATIC',
    'BCH', 'VET', 'EGLD', 'XLM', 'AXS',
    'XTZ', 
]
BASES = ['USDT']
SYMBOLS = [s + b for s in SYMBOLS for b in BASES]


@click.group()
def cli():
    pass

@cli.command()
@click.option('--symbol', '-s', type=str, default=None, help="asset name, e.g., BTCUSDT")
@click.option('--freq', '-f', required=True, type=click.Choice(['4h', '1h']), help="frquency to use")
@click.option('--res_dir', '-r', required=True, type=str, help="directory for outputs")
@click.option('--flag_filter', '-g', type=str, default=None, show_default=True, help='flag to use, mmi | mmi_ge | ang')
def best_ma_strategy(symbol, freq, res_dir, flag_filter):
    parts = res_dir.split('-')
    if freq not in parts:
        raise(ValueError('Mismatch found in freq setting and output dir'))
    if (flag_filter and not flag_filter in parts) or (not flag_filter and 'filter' in parts):
        raise(ValueError('Mismatch found in filter setting and output dir'))
    if symbol:
        symbols = [symbol]
    else:
        symbols = SYMBOLS
    BestMaStrategy(symbols, freq, res_dir, flag_filter)

@cli.command()
@click.option('--symbol', '-s', type=str, default=None, help='asset name, e.g., BTCUSDT')
@click.option('--date', '-d', type=str, help='the date when the results are generated')
@click.option('--res_dir', '-r', required=True, type=str, help='directory for outputs')
@click.option('--flag_filter', '-g', type=str, default=None, show_default=True, help='flag to use, mmi | mmi_ge | ang')
def check_ma_indicators(symbol, date, res_dir, flag_filter, ):
    parts = res_dir.split('-')
    if (flag_filter and not flag_filter in parts) or (not flag_filter and 'filter' in parts):
        raise(ValueError('Mismatch found in filter setting and output dir'))
    if not 'ma' in parts:
        raise(ValueError('Make sure the res_dir is for MA strategy'))
    if symbol:
        symbols = [symbol]
    else:
        symbols = SYMBOLS
    CheckMaIndicators(symbols, date, res_dir, flag_filter)  


@cli.command()
@click.option('--symbol', '-s', type=str, default=None, show_default=True, help="asset name, e.g., BTCUSDT")
@click.option('--freq', '-f', required=True, type=click.Choice(['4h', '1h']), help="frquency to use")
@click.option('--res_dir', '-r', required=True, type=str, help="directory for outputs")
@click.option('--flag_filter', '-g', type=str, default=None, show_default=True, help='filter to use, vol | ang')
@click.option('--flag_ts_stop', '-t', is_flag=True, help='ts_stop flag')
@click.option('--strategy', '-e', type=str, default='bo_rev', show_default=True, help='bo strategy, bo | bo_rev')
def best_bo_strategy(symbol, freq, res_dir, flag_filter, flag_ts_stop, strategy):
    parts = res_dir.split('-')
    if freq not in parts:
        raise(ValueError('Mismatch found in freq setting and output dir'))
    if (flag_filter and not flag_filter in parts) or (not flag_filter and 'filter' in parts):
        raise(ValueError('Mismatch found in filter setting and output dir'))
    if (strategy == 'bo_rev' and 'bo_rev' not in parts) or (strategy == 'bo' and 'bo' not in parts):
        raise(ValueError('Mismatch found in bo_revised setting and output dir'))
    if flag_ts_stop and 'ts_stop' not in parts:
        raise(ValueError('Mismatch found in ts_stop setting and output dir'))
    if symbol:
        symbols = [symbol]
    else:
        symbols = SYMBOLS
    BestBoStrategy(symbols, freq, res_dir, flag_filter, flag_ts_stop, strategy)

@cli.command()
@click.option('--symbol', '-s', type=str, default=None, help="asset name, e.g., BTCUSDT")
@click.option('--date', '-d', type=str, help='the date when the results are generated')
@click.option('--res_dir', '-r', required=True, type=str, help="directory for outputs")
@click.option('--flag_filter', '-g', type=str, default=None, show_default=True, help='filter to use, vol | ang')
@click.option('--strategy', '-e', type=str, default='bo_rev', show_default=True, help='bo strategy, bo | bo_rev')
def check_bo_indicators(symbol, date, res_dir, flag_filter, strategy):
    parts = res_dir.split('-')
    if (flag_filter and not flag_filter in parts) or (not flag_filter and 'filter' in parts):
        raise(ValueError('Mismatch found in filter setting and output dir'))
    if (strategy == 'bo_rev' and 'bo_rev' not in parts) or (strategy == 'bo' and 'bo' not in parts):
        raise(ValueError('Mismatch found in bo_revised setting and output dir'))
    if symbol:
        symbols = [symbol]
    else:
        symbols = SYMBOLS
    CheckBoIndicators(symbols, date, res_dir, flag_filter, strategy)  


@cli.command()
@click.option('--symbol', '-s', type=str, default=None, help="asset name, e.g., BTCUSDT")
def generate_report(symbol):
    if symbol:
        _generate_report(symbol)
    else:
        for symbol in SYMBOLS:
            _generate_report(symbol)


if __name__ == '__main__':
    cli()
