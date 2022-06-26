from itertools import product
from collections.abc import Iterable
import vectorbt as vbt
import pandas as pd
import numpy as np


def is_evalable(obj):
    try:
        eval(str(obj))
        return True
    except:
        return False


def remove_pd_object(d):
    ret = {}
    for n, v in d.items():
        if ((not isinstance(v, pd.Series)
            and not isinstance(v, pd.DataFrame)
            and not callable(v)
            and is_evalable(v)
            ) or isinstance(v, str)):
            ret[n] = v
    return ret


def enumerate_variables(variables):
    if not variables:
        return []

    enumeration_name = []
    enumeration_vars = []

    constant_d = {}

    for name, v in variables.items():
        if (
           isinstance(v, Iterable)
           and not isinstance(v, str)
           and not isinstance(v, pd.Series)
           and not isinstance(v, pd.DataFrame)
           ):
            enumeration_name.append(name)
            enumeration_vars.append(v)
        else:
            constant_d[name] = v

    variable_enumerations = [dict(**dict(zip(enumeration_name, ps)), **constant_d)
                             for ps in list(product(*enumeration_vars))]

    return variable_enumerations


def enumerate_signal(ohlcv, strategy, variables, ):
    entries = {}
    exits = {}

    fig = {}

    iteration = variables if len(variables) > 1 else variables
    for v in iteration:
        strategy.set_parameters(v)
        results = strategy.func(ohlcv)

        v = remove_pd_object(v)

        entries[str(v)], exits[str(v)] = results[0], results[1]
        if len(results) >= 3:
            fig = results[2]

    entries = pd.DataFrame(entries)
    exits = pd.DataFrame(exits)

    # setup columns
    param_names = list(eval(entries.columns[0]).keys())
    arrays = ([entries.columns.map(lambda s: eval(s)[p]) for p in param_names])
    tuples = list(zip(*arrays))
    if tuples:
        columns = pd.MultiIndex.from_tuples(tuples, names=param_names)
        exits.columns = columns
        entries.columns = columns
    return entries, exits, fig


def migrate_stop_vars(stop_vars):
    # to support OHLCSTX upgrade
    if 'ts_stop' in stop_vars:
        stop_vars['sl_stop'] = stop_vars['ts_stop']
        stop_vars['sl_trail'] = [s > 0 for s in stop_vars['ts_stop']]
    elif 'sl_stop' in stop_vars:
        stop_vars['sl_trail'] = [s < 0 for s in stop_vars['sl_stop']]
    return stop_vars


def stop_early(ohlcv, entries, exits, stop_vars, enumeration=True):
    if not stop_vars:
        return entries, exits

    # check for stop_vars
    length = -1
    stop_vars_set = {'sl_stop', 'ts_stop', 'tp_stop'}
    for s, slist in stop_vars.items():
        if s not in stop_vars_set:
            raise Exception(f'variable { s } is not one of the stop variables'
                            ': sl_stop, ts_stop, or tp_stop')
        if not isinstance(slist, Iterable):
            stop_vars[s] = [slist]

        if length == -1:
            length = len(stop_vars[s])

        if not enumeration and length != -1 and length != len(stop_vars[s]):
            raise Exception('lengths of the variables are not align: ',
                            str([len(stop_vars[s]) for s, slist in stop_vars.items()]))

    if enumeration:
        stop_vars = enumerate_variables(stop_vars)
        stop_vars = {key: [stop_vars[i][key] for i in range(len(stop_vars))] for key in stop_vars[0].keys()}

    # import pdb; pdb.set_trace()

    stop_vars = migrate_stop_vars(stop_vars)

    ohlcstx = vbt.OHLCSTX.run(
        entries,
        ohlcv['open'],
        ohlcv['high'],
        ohlcv['low'],
        ohlcv['close'],
        **stop_vars,
    )
    stop_exits = ohlcstx.exits

    nrepeat = int(len(stop_exits.columns) / len(entries.columns))
    if isinstance(stop_exits, pd.DataFrame):
        exits = exits.vbt.tile(nrepeat)
        entries = entries.vbt.tile(nrepeat)

    stop_exits = stop_exits.vbt | exits.values
    entries.columns = stop_exits.columns

    return entries, stop_exits
