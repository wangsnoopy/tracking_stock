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
import indicators as ind # Assuming you name your indicators file 'indicators.py'
from util import get_data # Assuming util.py is in the parent directory or PYTHONPATH is set

# --- Helper function for marketsim (adapt your marketsimcode.py) ---
# This version of compute_portvals directly accepts a trades DataFrame
# You should integrate this logic into your marketsimcode.py's compute_portvals
# or create a new function in it like compute_portvals_from_trades
def author():  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    :return: The GT username of the student  		  	   		 	 	 			  		 			 	 	 		 		 	
    :rtype: str  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    return "awang758" 


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

    unique_symbols = trades_df.columns.tolist()
    start_date = trades_df.index.min()
    end_date = trades_df.index.max()

    prices = get_data(unique_symbols, pd.date_range(start_date, end_date)).dropna()
    prices["Cash"] = 1.0 # Add cash column for calculations

    # Ensure prices are aligned with trades_df index, handling potential missing dates
    prices = prices.reindex(trades_df.index).ffill().bfill()
    prices['Cash'] = 1.0 # Ensure cash column exists after reindexing

    # Initialize trades DataFrame with all necessary columns (symbols + Cash)
    full_trades_df = pd.DataFrame(0.0, columns=prices.columns, index=prices.index)

    # Populate full_trades_df based on trades_df
    for symbol in unique_symbols:
        if symbol in trades_df.columns:
            full_trades_df[symbol] = trades_df[symbol]

    # Calculate cash changes due to trades, commission, and impact
    for date in trades_df.index:
        for symbol in unique_symbols:
            if trades_df.loc[date, symbol] != 0:
                shares = trades_df.loc[date, symbol]
                price = prices.loc[date, symbol]
                trade_value = shares * price
                cash_change = -trade_value - (commission + impact * abs(trade_value))
                full_trades_df.loc[date, 'Cash'] += cash_change

    holdings = pd.DataFrame(0.0, columns=full_trades_df.columns, index=full_trades_df.index)
    holdings.iloc[0] = full_trades_df.iloc[0]
    holdings.loc[holdings.index[0], "Cash"] += start_val

    for i in range(1, len(holdings)):
        holdings.iloc[i] = holdings.iloc[i-1] + full_trades_df.iloc[i]

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

def plot_normalized_data(df, title, ylabel, xlabel="Date", save_path="./images/"):
    plt.figure(figsize=(12, 7))
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"{save_path}{title.replace(' ', '_').replace('/', '')}.png")
    plt.close()


if __name__ == "__main__":
    # Define parameters as per project requirements
    symbol = "JPM"
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000

    # --- Part 1: Theoretically Optimal Strategy (TOS) ---
    print("--- Running Theoretically Optimal Strategy (TOS) ---")
    df_trades_tos = tos.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)

    # Use the adapted market simulator to get portfolio values for TOS
    # IMPORTANT: Ensure compute_portvals_from_trades is available in marketsimcode.py
    # or adapt marketsimcode.py's compute_portvals to accept DataFrame directly.
    portvals_tos = compute_portvals_from_trades(
        trades_df=df_trades_tos,
        start_val=sv,
        commission=0.00, # As per TOS requirements
        impact=0.00     # As per TOS requirements
    )

    # --- Benchmark Calculation ---
    # Benchmark: starting with $100,000 cash, investing in 1000 shares of JPM, and holding
    prices_jpm = get_data([symbol], pd.date_range(sd, ed), addSPY=False).dropna()
    prices_jpm = prices_jpm[[symbol]] # Ensure only JPM column
    
    # Create trades for benchmark (Buy 1000 shares on the first day, hold)
    df_trades_benchmark = pd.DataFrame(0.0, index=prices_jpm.index, columns=[symbol])
    df_trades_benchmark.iloc[0, 0] = 1000 # Buy 1000 shares on the first day

    # Compute portfolio value for benchmark (using the same commission and impact as TOS)
    portvals_benchmark = compute_portvals_from_trades(
        trades_df=df_trades_benchmark,
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
    plt.savefig("./images/TOS_vs_Benchmark.png")
    plt.close()
    print("Generated TOS vs. Benchmark plot: ./images/TOS_vs_Benchmark.png")

    # --- Compute and Print TOS and Benchmark Statistics ---
    cr_tos, adr_tos, sddr_tos, sr_tos = compute_portfolio_stats(portvals_tos['Portfolio Value'])
    cr_benchmark, adr_benchmark, sddr_benchmark, sr_benchmark = compute_portfolio_stats(portvals_benchmark['Portfolio Value'])

    print("\n--- Portfolio Statistics (to 6 decimal places) ---")
    print(f"{'Metric':<30} | {'Benchmark':<15} | {'Optimal Portfolio':<20}")
    print(f"{'-'*30} | {'-'*15} | {'-'*20}")
    print(f"{'Cumulative Return':<30} | {cr_benchmark:<15.6f} | {cr_tos:<20.6f}")
    print(f"{'Stdev of Daily Returns':<30} | {sddr_benchmark:<15.6f} | {sddr_tos:<20.6f}")
    print(f"{'Mean of Daily Returns':<30} | {adr_benchmark:<15.6f} | {adr_tos:<20.6f}")
    # Note: Sharpe Ratio is not explicitly requested in the table, but often calculated.
    # print(f"{'Sharpe Ratio':<30} | {sr_benchmark:<15.6f} | {sr_tos:<20.6f}")

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
    prices_for_indicators = prices_for_indicators[[symbol]] # Ensure only the target symbol

    # 1. Bollinger Bands Percentage (%B)
    bbp_values = ind.bollinger_bands_percentage(prices_for_indicators, window=20)
    
    # Re-calculate helper data for plotting within testproject.py for consistency
    sma_bb = ind.get_rolling_mean(prices_for_indicators[symbol], window=20)
    rstd_bb = ind.get_rolling_std(prices_for_indicators[symbol], window=20)
    upper_band = sma_bb + (2 * rstd_bb)
    lower_band = sma_bb - (2 * rstd_bb)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))
    ax1.plot(prices_for_indicators.index, prices_for_indicators[symbol], label=f'{symbol} Price', color='blue')
    ax1.plot(prices_for_indicators.index, sma_bb, label='SMA (20)', color='orange', linestyle='--')
    ax1.plot(prices_for_indicators.index, upper_band, label='Upper Band', color='red', linestyle=':')
    ax1.plot(prices_for_indicators.index, lower_band, label='Lower Band', color='green', linestyle=':')
    ax1.set_ylabel("Price")
    ax1.set_title(f"{symbol} Price with Bollinger Bands")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(bbp_values.index, bbp_values, label='Bollinger %B', color='purple')
    ax2.axhline(1.0, color='red', linestyle='--', label='Overbought (1.0)')
    ax2.axhline(0.0, color='green', linestyle='--', label='Oversold (0.0)')
    ax2.set_xlabel("Date")
    ax2.set_ylabel("%B Value")
    ax2.set_title("Bollinger Bands Percentage (%B)")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig("./images/Bollinger_Bands_Percentage.png")
    plt.close()
    print("Generated Bollinger Bands Percentage plot: ./images/Bollinger_Bands_Percentage.png")

    # Placeholder for other indicators (you will add 4 more here)
    # You will need to implement functions for these in indicators.py
    # and then call them and plot them similarly to Bollinger Bands Percentage.

    print("\n--- All tasks completed. Review generated plots in the 'images' folder and p6_results.txt. ---")