import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime as dt

def _unique_filepath(directory: str, basename: str, ext: str = ".png"):
    """
    Helper to generate a filepath that won’t overwrite:
      directory/basename_YYYYmmdd_HHMMSS.ext
    """
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{basename}_{ts}{ext}"
    return os.path.join(directory, filename)

def plot_total_return_bar(perf_df, saveto, top_n=10, figsize=(8,4)):
    """
    Plots a horizontal bar chart of total returns for the top_n signals.

    Parameters:
    - perf_df: DataFrame with columns ['signal_name', 'total_return_pct', ...], sorted by performance metric.
    - top_n:   Number of top signals to display.
    - figsize: Figure size tuple.
    """
    top_signals = perf_df.head(top_n)
    plt.figure(figsize=figsize)
    plt.barh(top_signals['signal_name'], top_signals['total_return_pct'], color='C1')
    plt.xlabel('Total Return (%)')
    plt.title(f'Top {top_n} Signals by Total Return')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    path = _unique_filepath(saveto, "top_returns", ".png")
    plt.savefig(path)
    plt.show()


def plot_equity_curve(signals_df, returns, saveto, signal_name=None, figsize=(10,4)):
    """
    Plots an equity curve for a single signal or an equal-weighted portfolio.

    Parameters:
    - signals_df: DataFrame of shape [T x N] with values in {-1,0,1} for each signal.
    - returns:    Series or array of asset returns indexed by timestamp.
    - signal_name: str or list. If str, plots that single signal. If list, plots an equal-weighted portfolio of those signals.
    - figsize:    Figure size tuple.
    """
    # prepare returns Series
    rets = returns if isinstance(returns, pd.Series) else pd.Series(returns, index=signals_df.index)

    if isinstance(signal_name, str):
        # single signal
        sig = signals_df[signal_name]
        strat_rets = sig * rets
        label = signal_name
        base = f"equity_{signal_name}"
    elif isinstance(signal_name, (list, tuple)):
        # equal-weighted portfolio
        subset = signals_df[signal_name]
        strat_rets = subset.multiply(rets, axis=0).mean(axis=1)
        label = 'Portfolio: ' + ','.join(signal_name)
        base = f"equity_portfolio"
    else:
        raise ValueError("signal_name must be a string or a list of strings")

    equity = (1 + strat_rets).cumprod()

    plt.figure(figsize=figsize)
    plt.plot(equity.index, equity, lw=2, label=label)
    plt.title(f'Equity Curve: {label} (×{equity.iloc[-1]:.2f})')
    plt.xlabel('Date')
    plt.ylabel('Growth of $1')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path = _unique_filepath(saveto, base, ".png")
    plt.savefig(path)
    plt.show()