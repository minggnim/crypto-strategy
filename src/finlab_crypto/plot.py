from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from . import chart


def plot_strategy(ohlcv, entries, exits, portfolio, fig_data, html=None):

    # format trade data
    txn = portfolio.positions.records
    txn['enter_time'] = ohlcv.iloc[txn.entry_idx].index.values
    txn['exit_time'] = ohlcv.iloc[txn.exit_idx].index.values

    # plot trade data
    mark_lines = []
    for name, t in txn.iterrows():
        x = [str(t.enter_time), str(t.exit_time)]
        y = [t.entry_price, t.exit_price]
        name = t.loc[['entry_price', 'exit_price', 'return']].to_string()
        mark_lines.append((name, x, y))

    # calculate overlap figures
    overlaps = {}
    if 'overlaps' in fig_data:
        overlaps = fig_data['overlaps']

    # calculate sub-figures
    figures = {}
    if 'figures' in fig_data:
        figures = fig_data['figures']

    figures['entries & exits'] = pd.DataFrame(
        {'entries': entries.squeeze(), 'exits': exits.squeeze()})
    figures['performance'] = portfolio.cumulative_returns()

    c, info = chart.chart(ohlcv, overlaps=overlaps,
                          figures=figures, markerlines=mark_lines,
                          start_date=ohlcv.index[-len(ohlcv)], end_date=ohlcv.index[-1])
    c.load_javascript()
    if html is not None:
        c.render(html)
    else:
        c.render()
        display(HTML(filename='render.html'))

    return


def plot_combination(portfolio, cscv_result=None, metric='final_value'):

    sns.set()
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(15, 4), sharey=False, sharex=False)
    fig.suptitle('Backtest Results')

    def heat_map(item, name1, name2, ax):
        if name1 != name2:
            sns.heatmap(item.reset_index().pivot(name1, name2)[0], cmap='magma_r', ax=ax)
        else:
            getattr(portfolio, item).groupby(name1).mean().plot(ax=ax)

    def best_n(portfolio, n):
        return getattr(portfolio, metric)().sort_values().tail(n).index

    best_10 = best_n(portfolio, 10)

    ax = (portfolio.cumulative_returns()[best_10] * 100).plot(ax=axes[0])
    ax.set(xlabel='time', ylabel='cumulative return (%)')

    axes[1].title.set_text('Drawdown (%)')
    for n, c in zip([5, 10, 20, 30], sns.color_palette("GnBu_d")):
        bests = best_n(portfolio, n)
        drawdown = portfolio.drawdown()[bests].min(axis=1)
        ax = drawdown.plot(linewidth=1, ax=axes[1])
        # ax.fill_between(drawdown.index, 0, drawdown * 100, alpha=0.2, color=c)
    ax.set(xlabel='time', ylabel='drawdown (%)')

    plt.show()

    items = ['final_value', 'sharpe_ratio', 'sortino_ratio']
    fig, axes = plt.subplots(1, len(items), figsize=(15, 3),
                             sharey=False, sharex=False, constrained_layout=False)
    fig.subplots_adjust(top=0.75)
    fig.suptitle('Partial Differentiation')

    final_value = portfolio.final_value()
    if isinstance(final_value.index, pd.MultiIndex):
        index_names = final_value.index.names
    else:
        index_names = [final_value.index.name]

    for i, item in enumerate(items):
        results = {}
        for name in index_names:
            s = getattr(portfolio, item)()
            s = s.replace([np.inf, -np.inf], np.nan)
            results[name] = s.groupby(name).mean()
        results = pd.DataFrame(results)
        axes[i].title.set_text(item)
        results.plot(ax=axes[i])

    if cscv_result is None:
        return

    results = cscv_result

    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             sharey=False, sharex=False, constrained_layout=False)
    fig.subplots_adjust(bottom=0.5)
    fig.suptitle('Combinatorially Symmetric Cross-validation')

    pbo_test = round(results['pbo_test'] * 100, 2)
    axes[0].title.set_text(f'Probability of overfitting: {pbo_test} %')
    axes[0].hist(x=[l for l in results['logits'] if l > -10000], bins='auto')
    axes[0].set_xlabel('Logits')
    axes[0].set_ylabel('Frequency')

    # performance degradation
    axes[1].title.set_text('Performance degradation')
    x, y = pd.DataFrame([results['R_n_star'], results['R_bar_n_star']]).dropna(axis=1).values
    sns.regplot(x, y, ax=axes[1])
    # axes[1].set_xlim(min(results['R_n_star']) * 1.2,max(results['R_n_star']) * 1.2)
    # axes[1].set_ylim(min(results['R_bar_n_star']) * 1.2,max(results['R_bar_n_star']) * 1.2)
    axes[1].set_xlabel('In-sample Performance')
    axes[1].set_ylabel('Out-of-sample Performance')

    # first and second Stochastic dominance
    axes[2].title.set_text('Stochastic dominance')
    if len(results['dom_df']) != 0:
        results['dom_df'].plot(ax=axes[2], secondary_y=['SD2'])
    axes[2].set_xlabel('Performance optimized vs non-optimized')
    axes[2].set_ylabel('Frequency')


def variable_visualization(portfolio):

    param_names = portfolio.cumulative_returns().columns.names
    dropdown1 = widgets.Dropdown(
        options=param_names,
        value=param_names[0],
        description='axis 1:',
        disabled=False,
    )
    dropdown2 = widgets.Dropdown(
        options=param_names,
        value=param_names[0],
        description='axis 2:',
        disabled=False,
    )

    performance_metric = [
        'final_value',
        'calmar_ratio', 'max_drawdown', 'sharpe_ratio',
        'downside_risk', 'omega_ratio', 'conditional_value_at_risk']

    performance_dropdwon = widgets.Dropdown(
        options=performance_metric,
        value=performance_metric[0],
        description='performance',
        disabled=False,
    )

    out = widgets.Output()

    def update(v):
        name1 = dropdown1.value
        name2 = dropdown2.value
        performance = performance_dropdwon.value

        with out:
            out.clear_output()
            if name1 != name2:
                df = (getattr(portfolio, performance)()
                      .reset_index().groupby([name1, name2]).mean()[0]
                      .reset_index().pivot(name1, name2)[0])

                df = df.replace([np.inf, -np.inf], np.nan)
                sns.heatmap(df)
            else:
                getattr(portfolio, performance)().groupby(name1).mean().plot()
            plt.show()

    dropdown1.observe(update, 'value')
    dropdown2.observe(update, 'value')
    performance_dropdwon.observe(update, 'value')
    drawdowns = widgets.VBox([
        performance_dropdwon,
        widgets.HBox([dropdown1, dropdown2])])
    display(drawdowns)
    display(out)
    update(0)