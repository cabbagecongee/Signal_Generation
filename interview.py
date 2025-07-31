import pandas as pd
import numpy as np
import warnings

"""
Your task is to extend this signal generation framework. You will start with a basic system that generates one simple signal (Momentum). Your goal is to add new, more sophisticated signals and analyze their performance.

Overall outline:

1. Familiarize yourself with the code and understand the structure of the Config, Performance, SignalFactory, and Backtester classes. As an overview, the Config class centralizes all key parameters, such as file paths, 
the annualization factor for metrics, and backtest settings, making the system easy to reconfigure. The Performance class contains all the formulas to calculate standard financial metrics like the Sharpe, Sortino, and 
Calmar ratios, which are used to judge a strategy's risk-adjusted return.
The SignalFactory is the core of the framework and takes in the raw price data and generated various trading signals (represented as -1 for short, 0 for neutral, and 1 for long). You will add your own signal-generating logic here. 
Finally, the Backtester class is the engine that takes the generated signals and the price series, calculates the hypothetical returns of each strategy, and uses the Performance class to create a final report and signal ranking.

2. Add New Signals: In the SignalFactory class, you will find a TODO section. Your primary task is to implement at least five new signal generation methods. For example, you can consider signals based on moving averages, 
cross asset relationships, or any other idea that you can come up with. This is purposefully left as open ended as possible. We want to see your creativity here in the signal development stage. To integrate your signals, 
define them in the SignalFactory class and then add them to the buid function so they are included in the backtest.
    - Right now, signals are simply tested over the entire dataset. As a further extension if you wish to pursue it, splitting the signal generation into train and test splits may be helpful to see if your signals generalize.
    (Do the top signals in the first period still do well out of sample?)

3. Run the script and analyze the results. The output CSV file will contain the performance metrics for all signals, including the ones you created. 
    - Analyze these results and discuss any interesting findings or areas of research
    - See how these signals could be combined to form a strong trading strategy. You can implement this using code or provide a written explanation and high-level overview.

4. (Optional) Enhance the Framework: For an even more rigorous submission, feel free to improve the framework following conventional computer science principles. There may be optimizations 
(vectorization of computations, parallelization, class refactoring, modularization) that you could make to the code to enhance its robustness. Additionally, this code may have some suboptimal implementation details, 
that while technically correct, aren't fully robust and could be improved. 
"""

# ───────────── Configuration ─────────────
class Config:
    """
    Configuration class for the signal generation and backtesting script.
    """
    # Path to the input CSV file. The file should have a 'date' column
    # and at least one price column (e.g., 'close').
    csv_path: str = "ES.csv"

    # Path to write the final performance analysis CSV.
    output_path: str = "candidate_signal_analysis.csv"

    # Annualization factor used for performance metrics like Sharpe Ratio.
    # (Default: 252 trading days * 7 intraday periods)
    annual_factor: int = 252 * 7

    # Number of initial rows to skip to ensure enough data for lookback calculations (e.g., momentum).
    warmup: int = 50

    # Index of the price column to be used for the backtest.
    # We will dynamically find the 'close' column, but you can override it here.
    asset_col_idx: int = None

    # Flag for whether to allow overnight returns. If False, returns from the close of one day
    # to the open of the next are set to zero.
    overnight_allowed: bool = False


# ───────────── Performance Metrics ─────────────
class Performance:
    """
    A collection of static methods to calculate common financial performance metrics.
    """
    @staticmethod
    def sharpe(rets: np.ndarray, ann: int) -> float:
        """Calculates the annualized Sharpe Ratio."""
        if len(rets) == 0:
            return 0.0
        
        total_return = np.prod(1 + rets) - 1
        ann_ret = (1 + total_return) ** (ann / len(rets)) - 1
        ann_std = np.std(rets, ddof=1) * np.sqrt(ann)
        
        if ann_std == 0:
            return 0.0 if ann_ret == 0 else np.inf * np.sign(ann_ret)
            
        return ann_ret / ann_std

    @staticmethod
    def sortino(rets: np.ndarray, ann: int) -> float:
        """Calculates the annualized Sortino Ratio."""
        if len(rets) == 0:
            return 0.0
            
        total_return = np.prod(1 + rets) - 1
        ann_ret = (1 + total_return) ** (ann / len(rets)) - 1
        
        # Calculate downside deviation
        downside_rets = np.minimum(rets, 0)
        downside_dev = np.sqrt(np.mean(downside_rets ** 2)) * np.sqrt(ann)
        
        if downside_dev == 0:
            return 0.0 if ann_ret == 0 else np.inf * np.sign(ann_ret)
            
        return ann_ret / downside_dev

    @staticmethod
    def calmar(rets: np.ndarray, ann: int) -> float:
        """Calculates the annualized Calmar Ratio."""
        if len(rets) == 0:
            return 0.0

        total_return = np.prod(1 + rets) - 1
        ann_ret = (1 + total_return) ** (ann / len(rets)) - 1
        
        # Calculate maximum drawdown
        equity_curve = np.cumprod(1 + rets)
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        max_dd = np.min(drawdowns)
        
        if max_dd == 0:
            return 0.0 if ann_ret == 0 else np.inf * np.sign(ann_ret)
            
        return ann_ret / abs(max_dd)


# ───────────── Signal Factory ─────────────
class SignalFactory:
    """
    This class is responsible for generating trading signals from the price data.
    """
    def __init__(self, df: pd.DataFrame, cfg: Config):
        self.cfg = cfg
        self.df = df.copy()
        
        # Ensure the index is a DatetimeIndex
        if not isinstance(self.df.index, pd.DatetimeIndex):
            # Assuming the first column is the date column
            self.df.iloc[:, 0] = pd.to_datetime(self.df.iloc[:, 0])
            self.df.set_index(self.df.columns[0], inplace=True)
        
        # Isolate the price data to be used for signal generation
        self.prices_df = self.df.iloc[:, 1:] # Use all columns except the original date column if it was present
        self.returns = self.prices_df.pct_change()

    # =========================================================================================
    # TODO: Please add new signal generation methods below this comment.
    #
    # Each method should return a pandas DataFrame where:
    #   - The index is the timestamp.
    #   - The columns are the unique signal names.
    #   - The values are -1 (short), 1 (long), or 0 (neutral).
    #
    # Remember to give your signals unique names to identify them in the final report.
    # Example: "SMA_CROSS|close|50|200"
    # =========================================================================================

    def momentum(self) -> pd.DataFrame:
        """
        Example Signal: Generates momentum signals.
        A long signal (1) is generated if the price has increased over the lookback period `m`.
        A short signal (-1) is generated if the price has decreased.
        """
        parts = []
        for m in range(2, self.cfg.warmup): # Use a range of lookback periods
            # Signal is 1 if price is higher than `m` periods ago, -1 otherwise
            diff = (self.prices_df > self.prices_df.shift(m))
            dfm = diff.astype(int) * 2 - 1
            dfm.columns = [f"MOM|{c}|{m}" for c in dfm.columns]
            parts.append(dfm)
        return pd.concat(parts, axis=1)
    
    def simple_moving_average(self) -> pd.DataFrame:
        # Potentially implement a simple moving average signal
        pass

    def your_signal(self) -> pd.DataFrame:
        # Implement any other signals of your choice
        pass

    def build(self) -> pd.DataFrame:
        """
        This method constructs all signals, combines them into a single DataFrame,
        and prepares them for the backtester.
        """
        # --- Call your new signal methods here and add them to the list ---
        # For example:
        # my_new_signal = self.my_signal_method()
        # signal_list = [self.momentum(), my_new_signal]
        
        signal_list = [self.momentum()]
        
        # Combine all signals into one DataFrame
        all_signals = pd.concat(signal_list, axis=1)
        
        # Clean up the combined signals
        # 1. Remove rows with NaNs, which usually occur at the start due to lookbacks.
        # 2. Shift signals by 1 to prevent lookahead bias (i.e., we trade on the next bar's open).
        all_signals = all_signals.iloc[self.cfg.warmup:].dropna(axis=1, how='all')
        all_signals = all_signals.shift(1).dropna()
        
        return all_signals.astype('int8')


# ───────────── Backtester ─────────────
class Backtester:
    """
    This class takes the generated signals and evaluates their performance over the full dataset.
    """
    def __init__(self, signals: pd.DataFrame, prices: pd.Series, cfg: Config):
        self.signals_df = signals
        self.cfg = cfg

        # Align the price series with the signal dates
        common_index = self.signals_df.index.intersection(prices.index)
        self.signals_df = self.signals_df.loc[common_index]
        self.prices = prices.loc[common_index]

        # Calculate strategy returns
        rets = self.prices.pct_change().fillna(0)

        # Handle overnight returns based on config
        if not cfg.overnight_allowed:
            dates = self.prices.index.to_series()
            is_overnight = dates.dt.day != dates.shift(1).dt.day
            rets[is_overnight] = 0
            
        self.returns = rets.values
        
        # Remove the first row which has a 0 return from pct_change()
        self.signals_df = self.signals_df.iloc[1:]
        self.returns = self.returns[1:]


    def run_full_period_analysis(self) -> pd.DataFrame:
        """
        Evaluates each signal over the entire time period and returns a performance report.
        """
        n_signals = self.signals_df.shape[1]
        if n_signals == 0:
            print("No signals to process.")
            return pd.DataFrame()

        print(f"Backtesting {n_signals} signals over the full period...")
        
        results = []
        for signal_name in self.signals_df.columns:
            # Get the signal vector (-1, 0, 1)
            signal_vector = self.signals_df[signal_name].values
            
            # Calculate the returns of the strategy: signal * asset_return
            strategy_returns = signal_vector * self.returns
            
            # Calculate performance metrics
            total_return = (np.prod(1 + strategy_returns)) * 100
            sharpe_ratio = Performance.sharpe(strategy_returns, self.cfg.annual_factor)
            sortino_ratio = Performance.sortino(strategy_returns, self.cfg.annual_factor)
            calmar_ratio = Performance.calmar(strategy_returns, self.cfg.annual_factor)
            
            # Store results
            results.append({
                'signal_name': signal_name,
                'total_return_pct': total_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio
            })

        # Create and sort the results DataFrame
        results_df = pd.DataFrame(results)
        results_df.sort_values(by='sharpe_ratio', ascending=False, inplace=True)
        
        return results_df.reset_index(drop=True)

# ───────────── Main Execution ─────────────
if __name__ == "__main__":
    print("Starting signal generation and backtesting process...")
    
    cfg = Config()
    
    try:
        # --- Load Data ---
        df = pd.read_csv(cfg.csv_path)
        print(f"Loaded data from '{cfg.csv_path}' with shape: {df.shape}")

        # Find the 'close' price column to use for the backtest
        if cfg.asset_col_idx is None:
            try:
                # Find the column named 'close'
                close_col_name = next(col for col in df.columns if 'close' in col.lower())
                cfg.asset_col_idx = df.columns.get_loc(close_col_name)
                print(f"Using '{close_col_name}' column for backtesting.")
            except StopIteration:
                raise ValueError("Could not find a 'close' column in the data. Please set 'asset_col_idx' in the Config.")

        # --- Generate Signals ---
        factory = SignalFactory(df, cfg)
        signals = factory.build()
        print(f"Generated {signals.shape[1]} unique signals.")
        
        # Select the specific price series for the backtest
        prices_for_backtest = df.iloc[:, cfg.asset_col_idx]
        # Make sure the price series has the date as its index
        prices_for_backtest.index = pd.to_datetime(df.iloc[:, 0])

        # --- Run Backtest ---
        backtester = Backtester(signals, prices_for_backtest, cfg)
        performance_report = backtester.run_full_period_analysis()
        
        # --- Output Results ---
        if not performance_report.empty:
            print("\n--- Top 10 Signals by Sharpe Ratio ---")
            print(performance_report.head(10).to_string())
            
            performance_report.to_csv(cfg.output_path, index=False)
            print(f"\nFull performance report saved to: {cfg.output_path}")
        else:
            print("No performance report generated.")

    except FileNotFoundError:
        print(f"ERROR: The data file was not found at '{cfg.csv_path}'.")
        print("Please ensure the CSV file is in the correct directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
