# grid_search.py

import pandas as pd
import numpy as np
from itertools import product
from interview import Config, SignalFactory, Backtester  # adjust this import

def grid_search_signal(df, cfg, prices, signal_name, param_grid):
    """
    Runs a grid-search over one SignalFactory method.
    Returns a DataFrame of params + performance.
    """
    factory = SignalFactory(df, cfg)
    records = []

    # loop over every combination of grid params
    for combo in product(*param_grid.values()):
        # build a kwargs dict for this combo
        kwargs = dict(zip(param_grid.keys(), combo))
        # call the signal method with these named args
        sig_df = getattr(factory, signal_name)(**kwargs)

        # if the method returns multiple columns, pick the first
        single = sig_df.iloc[:, :1]

        bt = Backtester(single.shift(1).dropna(), prices, cfg)
        perf = bt.run_full_period_analysis()
        if perf.empty: 
            continue

        top = perf.iloc[0]
        records.append({**kwargs,
                        'signal': signal_name,
                        'sharpe':   top.sharpe_ratio,
                        'sortino':  top.sortino_ratio,
                        'calmar':   top.calmar_ratio})

    return pd.DataFrame(records).sort_values('sharpe', ascending=False)


if __name__ == "__main__":
    # --- load data & config exactly as in your main script ---
    cfg = Config()
    df  = pd.read_csv(cfg.csv_path)
    df.iloc[:,0] = pd.to_datetime(df.iloc[:,0])
    df.set_index(df.columns[0], inplace=True)

    # pick your backtest price series
    close_col = next(c for c in df.columns if 'close' in c.lower())
    prices = df[close_col]

    # define parameter grids for each signal
    grids = {
        'moving_average_crossover': {
            'price_col': ['close'],      # only testing close here
            'fast':      [5, 10, 20],
            'slow':      [50, 100, 150],
        },
        'rsi_reversion': {
            'period':      [14, 21, 30],
            'upper_thresh':[70, 80],
            'lower_thresh':[30, 20],
        },
        'bollinger_band_reversion': {
            'period': [20, 50, 100],
            'std':    [1.5, 2.0, 2.5],
        },
        'roc_crossover': {
            'p': [5, 10, 20, 30],
        },
        # etc. add other methods & their grids here...
    }

    # run grid searches
    all_results = []
    for signal_name, grid in grids.items():
        print(f"\n>> Grid-searching '{signal_name}' …")
        df_res = grid_search_signal(df, cfg, prices, signal_name, grid)
        print(df_res.head(5).to_string(index=False))
        all_results.append(df_res.head(5))

    # Combine top findings
    summary = pd.concat(all_results, ignore_index=True)
    summary.to_csv("grid_search_summary.csv", index=False)
    print("\nSaved top-5 results for each to grid_search_summary.csv")
