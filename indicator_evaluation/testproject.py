'''
Use only the data provided for this course. You are not allowed to import external data.  
Add an author() function to each file.  
For your report, use only the symbol JPM.  
Use the time period January 1, 2008, to December 31, 2009.
Starting cash is $100,000.  
'''
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import TheoreticallyOptimalStrategy as tos
import indicators as ind
# Make sure your marketsimcode.py has a function like compute_portvals_from_trades
# or its main compute_portvals function accepts a DataFrame directly.
# For this example, I'll assume you copy-pasted the helper function into marketsimcode.py
# or adapted its primary function.
import marketsimcode as msc # Assuming your marketsim code is in marketsimcode.py

from util import get_data # Assuming util.py is in the parent directory or PYTHONPATH is set
import os # Import os module to create directory

# --- Helper function for marketsim (adapt your marketsimcode.py) ---
# This version of compute_portvals directly accepts a trades DataFrame
# You should integrate this logic into your marketsimcode.py's compute_portvals
# or create a new function in it like compute_portvals_from_trades
def compute_portvals_from_trades(trades_df, start_val=100000, commission=0.0, impact=0.0):
    """
    Computes the portfolio values directly from a trades DataFrame.
    This is an adapted version of compute_portvals for Project 6 requirements.
    """
    if trades_df.empty:
        # If no trades, just return initial cash value over the period
        dates = pd.date_range(trades_df.index.min(), trades_df.index.max())
        portvals = pd.DataFrame(index=dates, data=start_val, columns=['Portfolio Value'])
        return portvals

    # Ensure trades_df has at least one column (the symbol)
    if trades_df.columns.empty:
        raise ValueError("trades_df must contain at least one symbol column.")
    
    unique_symbols = trades_df.columns.tolist()
    start_date = trades_df.index.min()
    end_date = trades_df.index.max()

    # Get prices for all unique symbols, handling potential SPY addition
    prices = get_data(unique_symbols, pd.date_range(start_date, end_date), addSPY=False).dropna()
    
    # Reindex prices to match trades_df index to ensure consistent dates
    prices = prices.reindex(trades_df.index).ffill().bfill()
    prices['Cash'] = 1.0 # Add cash column for calculations


    # Initialize a full trades DataFrame with all necessary columns (symbols + Cash)
    # This is where the actual cash changes from trades will be recorded
    full_trades_df = pd.DataFrame(0.0, columns=prices.columns, index=prices.index)

    # Populate full_trades_df based on trades_df and apply costs
    for date in trades_df.index:
        for symbol in unique_symbols:
            if symbol in trades_df.columns and trades_df.loc[date, symbol] != 0: # Check if symbol exists in trades_df for this date
                shares = trades_df.loc[date, symbol]
                price = prices.loc[date, symbol]

                # Update the shares for the symbol
                full_trades_df.loc[date, symbol] += shares

                # Calculate cash change including commission and impact
                trade_value = shares * price
                cash_change = -trade_value - (commission + impact * abs(trade_value))
                full_trades_df.loc[date, 'Cash'] += cash_change

    # Calculate holdings over time
    holdings = pd.DataFrame(0.0, columns=full_trades_df.columns, index=full_trades_df.index)
    holdings.iloc[0] = full_trades_df.iloc[0] # Apply first day's trades
    holdings.loc[holdings.index[0], "Cash"] += start_val # Add initial cash

    for i in range(1, len(holdings)):
        holdings.iloc[i] = holdings.iloc[i-1] + full_trades_df.iloc[i]

    # Calculate portfolio values
    values = prices * holdings
    port_vals = values.sum(axis=1)

    return pd.DataFrame(port_vals, columns=['Portfolio Value'])


def compute_portfolio_stats(port_vals, daily_rf=0):
    """
    Computes the cumulative return, average daily return, standard deviation of daily returns,
    and Sharpe Ratio for a given portfolio value series.
    """
    daily_returns = port_vals.pct_change().dropna()
    cumulative_return = (port_vals.iloc[-1] / port_vals.iloc[0]) - 1
    avg_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std(ddof=1) # Use ddof=1 for sample standard deviation
    sharpe_ratio = np.sqrt(252) * ((avg_daily_return - daily_rf) / std_daily_return)
    return cumulative_return, avg_daily_return, std_daily_return, sharpe_ratio

def plot_indicator_with_signals(price_series, indicator_series, title, indicator_label,
                                buy_signals=None, sell_signals=None,
                                overbought_level=None, oversold_level=None,
                                save_path="./images/"): # Re-added save_path
    """
    Plots the stock price and an indicator, along with buy/sell signals.
    Saves the plot to a file.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))

    # Plot 1: Price with Buy/Sell Signals
    ax1.plot(price_series.index, price_series, label=f'{price_series.name} Price', color='blue')
    if buy_signals is not None and not buy_signals.empty:
        ax1.scatter(buy_signals, price_series.loc[buy_signals], # Corrected from .index
                    marker='^', color='g', s=100, label='Buy Signal', alpha=0.7)
    if sell_signals is not None and not sell_signals.empty:
        ax1.scatter(sell_signals, price_series.loc[sell_signals], # Corrected from .index
                    marker='v', color='r', s=100, label='Sell Signal', alpha=0.7)
    ax1.set_ylabel("Price")
    ax1.set_title(f"{price_series.name} Price with {indicator_label} Signals")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Indicator Values
    ax2.plot(indicator_series.index, indicator_series, label=indicator_label, color='purple')
    if overbought_level is not None:
        ax2.axhline(overbought_level, color='red', linestyle='--', label='Overbought')
    if oversold_level is not None:
        ax2.axhline(oversold_level, color='green', linestyle='--', label='Oversold')

    ax2.set_xlabel("Date")
    ax2.set_ylabel(indicator_label)
    ax2.set_title(f"{indicator_label} Values")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    # Save the plot with the title as the filename
    filename = title.replace(' ', '_').replace('/', '').replace('%', 'pct') + ".png" # Added .replace('%', 'pct') for valid filename
    plt.savefig(f"{save_path}{filename}")
    plt.close() # Close the plot figure after saving


if __name__ == "__main__":
    # Define parameters as per project requirements
    symbol = "JPM"
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000

    # Ensure images directory exists
    if not os.path.exists("images"):
        os.makedirs("images")

    # --- Figure 1: TOS (Daily Portfolio Value) ---
    print("--- Running Theoretically Optimal Strategy (TOS) ---")
    df_trades_tos = tos.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)

    # Use the adapted market simulator to get portfolio values for TOS
    portvals_tos = compute_portvals_from_trades( # Using the local helper. Adapt marketsimcode.py for actual use.
        trades_df=df_trades_tos[[symbol]].fillna(0), # Ensure only symbol column and handle NaNs from TOS
        start_val=sv,
        commission=0.00,
        impact=0.00
    )

    # --- Benchmark Calculation ---
    # Benchmark: starting with $100,000 cash, investing in 1000 shares of JPM, and holding
    prices_jpm_benchmark = get_data([symbol], pd.date_range(sd, ed), addSPY=False).dropna()
    prices_jpm_benchmark = prices_jpm_benchmark[[symbol]] # Ensure only JPM column
    
    # Create trades for benchmark (Buy 1000 shares on the first day, hold)
    df_trades_benchmark = pd.DataFrame(0.0, index=prices_jpm_benchmark.index, columns=[symbol])
    df_trades_benchmark.iloc[0, 0] = 1000 # Buy 1000 shares on the first day

    # Compute portfolio value for benchmark (using the same commission and impact as TOS)
    portvals_benchmark = compute_portvals_from_trades( # Using the local helper
        trades_df=df_trades_benchmark[[symbol]].fillna(0), # Ensure only symbol column and handle NaNs
        start_val=sv,
        commission=0.00,
        impact=0.00
    )

    # --- Normalize and Plot TOS vs. Benchmark ---
    normalized_portvals_tos = portvals_tos / portvals_tos.iloc[0]
    normalized_portvals_benchmark = portvals_benchmark / portvals_benchmark.iloc[0]

    plot_df_tos_benchmark = pd.DataFrame(index=normalized_portvals_tos.index)
    plot_df_tos_benchmark['Benchmark'] = normalized_portvals_benchmark['Portfolio Value']
    plot_df_tos_benchmark['Optimal Portfolio'] = normalized_portvals_tos['Portfolio Value']

    plt.figure(figsize=(12, 7))
    ax = plot_df_tos_benchmark.plot(title="Theoretically Optimal Strategy vs. Benchmark (JPM)", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Portfolio Value")
    ax.plot(plot_df_tos_benchmark.index, plot_df_tos_benchmark['Benchmark'], color='purple', label='Benchmark')
    ax.plot(plot_df_tos_benchmark.index, plot_df_tos_benchmark['Optimal Portfolio'], color='red', label='Optimal Portfolio')
    ax.grid(True)
    ax.legend(loc="best")
    plt.tight_layout()
    # Save Figure 1
    plt.savefig("./images/TOS_Daily_Portfolio_Value.png")
    plt.close() # Close the figure
    print("Generated Figure 1: TOS_Daily_Portfolio_Value.png")

    # --- Compute and Print TOS and Benchmark Statistics ---
    cr_tos, adr_tos, sddr_tos, sr_tos = compute_portfolio_stats(portvals_tos['Portfolio Value'])
    cr_benchmark, adr_benchmark, sddr_benchmark, sr_benchmark = compute_portfolio_stats(portvals_benchmark['Portfolio Value'])

    print("\n--- Portfolio Statistics (to 6 decimal places) ---")
    print(f"{'Metric':<30} | {'Benchmark':<15} | {'Optimal Portfolio':<20}")
    print(f"{'-'*30} | {'-'*15} | {'-'*20}")
    print(f"{'Cumulative Return':<30} | {cr_benchmark:<15.6f} | {cr_tos:<20.6f}")
    print(f"{'Stdev of Daily Returns':<30} | {sddr_benchmark:<15.6f} | {sddr_tos:<20.6f}")
    print(f"{'Mean of Daily Returns':<30} | {adr_benchmark:<15.6f} | {adr_tos:<20.6f}")

    # Optionally write to a results file
    with open("p6_results.txt", "w") as f:
        f.write(f"Date Range: {sd} to {ed}\n")
        f.write("\n--- Portfolio Statistics ---\n")
        f.write(f"{'Metric':<30} | {'Benchmark':<15} | {'Optimal Portfolio':<20}\n")
        f.write(f"{'-'*30} | {'-'*15} | {'-'*20}\n")
        f.write(f"{'Cumulative Return':<30} | {cr_benchmark:<15.6f} | {cr_tos:<20.6f}\n")
        f.write(f"{'Stdev of Daily Returns':<30} | {sddr_benchmark:<15.6f} | {sddr_tos:<20.6f}\n")
        f.write(f"{'Mean of Daily Returns':<30} | {adr_benchmark:<15.6f} | {adr_tos:<20.6f}\n")
    print("TOS statistics written to p6_results.txt")


    # --- Part 2: Technical Indicators ---
    print("\n--- Running Technical Indicators ---")
    prices_for_indicators = get_data([symbol], pd.date_range(sd, ed), addSPY=False).dropna()
    prices_for_indicators = prices_for_indicators[[symbol]] # Ensure only the target symbol column

    # --- Figure 4: %B (JPM price with buy/sell signal from %B crossover) ---
    print("Generating Figure 4: %B plot...")
    bbp_values = ind.bollinger_bands_percentage(prices_for_indicators, window=20)
    
    # Generate signals for %B
    bbp_buy_signals = bbp_values[bbp_values < 0].index
    bbp_sell_signals = bbp_values[bbp_values > 1].index

    plot_indicator_with_signals(
        price_series=prices_for_indicators[symbol],
        indicator_series=bbp_values,
        title="JPM Price with Bollinger %B and Trade Signals",
        indicator_label="Bollinger %B",
        buy_signals=bbp_buy_signals,
        sell_signals=bbp_sell_signals,
        overbought_level=1.0,
        oversold_level=0.0
    )
    print("Generated Figure 4: JPM_Price_with_Bollinger_pctB_and_Trade_Signals.png")


    # --- Figure 2: CCI (JPM Price with buy/sell signal from CCI crossover) ---
    print("Generating Figure 2: CCI plot...")
    cci_values = ind.cci(prices_for_indicators, window=20)
    
    # Generate signals for CCI (e.g., crossover +/- 100 for buy/sell)
    cci_buy_signals = prices_for_indicators[(cci_values.shift(1) < -100) & (cci_values >= -100)].index
    cci_sell_signals = prices_for_indicators[(cci_values.shift(1) > 100) & (cci_values <= 100)].index

    plot_indicator_with_signals(
        price_series=prices_for_indicators[symbol],
        indicator_series=cci_values,
        title="JPM Price with CCI and Trade Signals",
        indicator_label="CCI",
        buy_signals=cci_buy_signals,
        sell_signals=cci_sell_signals,
        overbought_level=100,
        oversold_level=-100
    )
    print("Generated Figure 2: JPM_Price_with_CCI_and_Trade_Signals.png")


    # --- Figure 3: MACD (JPM price with buy/sell signal from MACD histogram crossover) ---
    print("Generating Figure 3: MACD Histogram plot...")
    macd_hist_values = ind.macd_histogram(prices_for_indicators, fast_period=12, slow_period=26, signal_period=9)

    # Generate signals for MACD Histogram (crossover zero line)
    macd_buy_signals = prices_for_indicators[(macd_hist_values.shift(1) < 0) & (macd_hist_values >= 0)].index
    macd_sell_signals = prices_for_indicators[(macd_hist_values.shift(1) > 0) & (macd_hist_values <= 0)].index

    plot_indicator_with_signals(
        price_series=prices_for_indicators[symbol],
        indicator_series=macd_hist_values,
        title="JPM Price with MACD Histogram and Trade Signals",
        indicator_label="MACD Histogram",
        buy_signals=macd_buy_signals,
        sell_signals=macd_sell_signals,
        overbought_level=0, # The zero line acts as a "crossover" level
        oversold_level=0
    )
    print("Generated Figure 3: JPM_Price_with_MACD_Histogram_and_Trade_Signals.png")


    # --- Figure 5: RSI (JPM price with buy/sell signall from RSI crossover) ---
    print("Generating Figure 5: RSI plot...")
    rsi_values = ind.rsi(prices_for_indicators, window=14)

    # Generate signals for RSI (crossover overbought/oversold levels)
    rsi_buy_signals = prices_for_indicators[(rsi_values.shift(1) < 30) & (rsi_values >= 30)].index
    rsi_sell_signals = prices_for_indicators[(rsi_values.shift(1) > 70) & (rsi_values <= 70)].index

    plot_indicator_with_signals(
        price_series=prices_for_indicators[symbol],
        indicator_series=rsi_values,
        title="JPM Price with RSI and Trade Signals",
        indicator_label="RSI",
        buy_signals=rsi_buy_signals,
        sell_signals=rsi_sell_signals,
        overbought_level=70,
        oversold_level=30
    )
    print("Generated Figure 5: JPM_Price_with_RSI_and_Trade_Signals.png")
    
    print("\n--- All tasks completed. Review generated plots in the 'images' folder and p6_results.txt. ---")