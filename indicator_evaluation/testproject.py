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
import marketsimcode as msc

from util import get_data
import os

def compute_portvals_from_trades(trades_df, start_val=100000, commission=0.0, impact=0.0):
    if trades_df.empty:
        dates = pd.date_range(trades_df.index.min(), trades_df.index.max())
        portvals = pd.DataFrame(index=dates, data=start_val, columns=['Portfolio Value'])
        return portvals

    if trades_df.columns.empty:
        raise ValueError("trades_df must contain at least one symbol column.")
    
    unique_symbols = trades_df.columns.tolist()
    start_date = trades_df.index.min()
    end_date = trades_df.index.max()

    prices = get_data(unique_symbols, pd.date_range(start_date, end_date), addSPY=False).dropna()
    
    prices = prices.reindex(trades_df.index).ffill().bfill()
    prices['Cash'] = 1.0

    full_trades_df = pd.DataFrame(0.0, columns=prices.columns, index=prices.index)

    for date in trades_df.index:
        for symbol in unique_symbols:
            if symbol in trades_df.columns and trades_df.loc[date, symbol] != 0:
                shares = trades_df.loc[date, symbol]
                price = prices.loc[date, symbol]

                full_trades_df.loc[date, symbol] += shares

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
    daily_returns = port_vals.pct_change().dropna()
    cumulative_return = (port_vals.iloc[-1] / port_vals.iloc[0]) - 1
    avg_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std(ddof=1) 
    sharpe_ratio = np.sqrt(252) * ((avg_daily_return - daily_rf) / std_daily_return)
    return cumulative_return, avg_daily_return, std_daily_return, sharpe_ratio

def plot_indicator_with_signals(price_series, indicator_series, title, indicator_label,
                                buy_signals=None, sell_signals=None,
                                overbought_level=None, oversold_level=None,
                                save_path="./images/"):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))

    ax1.plot(price_series.index, price_series, label=f'{price_series.name} Price', color='blue')
    if buy_signals is not None and not buy_signals.empty:
        ax1.scatter(buy_signals, price_series.loc[buy_signals],
                    marker='^', color='g', s=100, label='Buy Signal', alpha=0.7)
    if sell_signals is not None and not sell_signals.empty:
        ax1.scatter(sell_signals, price_series.loc[sell_signals],
                    marker='v', color='r', s=100, label='Sell Signal', alpha=0.7)
    ax1.set_ylabel("Price")
    ax1.set_title(f"{price_series.name} Price with {indicator_label} Signals")
    ax1.legend()
    ax1.grid(True)

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
    filename = title.replace(' ', '_').replace('/', '').replace('%', 'pct') + ".png"
    plt.savefig(f"{filename}")
    plt.close()


if __name__ == "__main__":
    symbol = "JPM"
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000

    df_trades_tos = tos.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)

    portvals_tos = compute_portvals_from_trades(
        trades_df=df_trades_tos[[symbol]].fillna(0),
        start_val=sv,
        commission=0.00,
        impact=0.00
    )

    prices_jpm_benchmark = get_data([symbol], pd.date_range(sd, ed), addSPY=False).dropna()
    prices_jpm_benchmark = prices_jpm_benchmark[[symbol]]
    
    df_trades_benchmark = pd.DataFrame(0.0, index=prices_jpm_benchmark.index, columns=[symbol])
    df_trades_benchmark.iloc[0, 0] = 1000

    portvals_benchmark = compute_portvals_from_trades(
        trades_df=df_trades_benchmark[[symbol]].fillna(0),
        start_val=sv,
        commission=0.00,
        impact=0.00
    )

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
    plt.savefig("TOS_Daily_Portfolio_Value.png")
    plt.close()

    cr_tos, adr_tos, sddr_tos, sr_tos = compute_portfolio_stats(portvals_tos['Portfolio Value'])
    cr_benchmark, adr_benchmark, sddr_benchmark, sr_benchmark = compute_portfolio_stats(portvals_benchmark['Portfolio Value'])

    print(f"{'Metric':<30} | {'Benchmark':<15} | {'Optimal Portfolio':<20}")
    print(f"{'-'*30} | {'-'*15} | {'-'*20}")
    print(f"{'Cumulative Return':<30} | {cr_benchmark:<15.6f} | {cr_tos:<20.6f}")
    print(f"{'Stdev of Daily Returns':<30} | {sddr_benchmark:<15.6f} | {sddr_tos:<20.6f}")
    print(f"{'Mean of Daily Returns':<30} | {adr_benchmark:<15.6f} | {adr_tos:<20.6f}")

    with open("p6_results.txt", "w") as f:
        f.write(f"Date Range: {sd} to {ed}\n")
        f.write("\n--- Portfolio Statistics ---\n")
        f.write(f"{'Metric':<30} | {'Benchmark':<15} | {'Optimal Portfolio':<20}\n")
        f.write(f"{'-'*30} | {'-'*15} | {'-'*20}\n")
        f.write(f"{'Cumulative Return':<30} | {cr_benchmark:<15.6f} | {cr_tos:<20.6f}\n")
        f.write(f"{'Stdev of Daily Returns':<30} | {sddr_benchmark:<15.6f} | {sddr_tos:<20.6f}\n")
        f.write(f"{'Mean of Daily Returns':<30} | {adr_benchmark:<15.6f} | {adr_tos:<20.6f}\n")

    prices_for_indicators = get_data([symbol], pd.date_range(sd, ed), addSPY=False).dropna()
    prices_for_indicators = prices_for_indicators[[symbol]]

    bbp_values = ind.bollinger_bands_percentage(prices_for_indicators, window=20)
    
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

    cci_values = ind.cci(prices_for_indicators, window=20)
    
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

    macd_hist_values = ind.macd_histogram(prices_for_indicators, fast_period=12, slow_period=26, signal_period=9)

    macd_buy_signals = prices_for_indicators[(macd_hist_values.shift(1) < 0) & (macd_hist_values >= 0)].index
    macd_sell_signals = prices_for_indicators[(macd_hist_values.shift(1) > 0) & (macd_hist_values <= 0)].index

    plot_indicator_with_signals(
        price_series=prices_for_indicators[symbol],
        indicator_series=macd_hist_values,
        title="JPM Price with MACD Histogram and Trade Signals",
        indicator_label="MACD Histogram",
        buy_signals=macd_buy_signals,
        sell_signals=macd_sell_signals,
        overbought_level=0,
        oversold_level=0
    )

    rsi_values = ind.rsi(prices_for_indicators, window=14)

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

    prices_for_indicators = get_data([symbol], pd.date_range(sd, ed), addSPY=False).dropna()
    prices_for_indicators = prices_for_indicators[[symbol]]

    psr_values = ind.price_sma_ratio(prices_for_indicators, window=20)
    psr_buy_signals = psr_values[psr_values < 1].index
    psr_sell_signals = psr_values[psr_values > 1.2].index

    plot_indicator_with_signals(
        price_series=prices_for_indicators[symbol],
        indicator_series=psr_values,
        title="JPM Price with Price/SMA Ratio and Trade Signals",
        indicator_label="Price/SMA Ratio",
        buy_signals=psr_buy_signals,
        sell_signals=psr_sell_signals,
        overbought_level=1.2,
        oversold_level=1.0
    )