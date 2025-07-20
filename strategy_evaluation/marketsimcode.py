import pandas as pd
import numpy as np
from util import get_data

def author():
    return 'awang758'

def compute_portvals(trades, start_val=100000, commission=9.95, impact=0.005):
    """
    Computes the portfolio values given a trades DataFrame.

    Parameters:
    trades: DataFrame with trades (positive for buy, negative for sell),
            indexed by date, column is the stock symbol
    start_val: Starting cash
    commission: Fixed cost per transaction
    impact: Market impact as a fraction of price

    Returns:
    portvals: DataFrame with portfolio value on each day
    """
    dates = trades.index
    symbol = trades.columns[0]
    prices = get_data([symbol], dates[0], dates[-1])
    prices = prices[[symbol]].ffill().bfill()

    cash = pd.Series(index=prices.index, data=start_val)
    holdings = trades.copy()
    holdings.values[:] = 0

    for i in range(len(prices)):
        date = prices.index[i]
        if i > 0:
            holdings.iloc[i] = holdings.iloc[i - 1]
            cash.iloc[i] = cash.iloc[i - 1]

        trade = trades.loc[date, symbol]
        if trade != 0:
            price = prices.loc[date, symbol]
            cost = trade * price
            total_fee = commission + abs(trade) * price * impact
            cash.iloc[i] -= cost + total_fee
            holdings.iloc[i, 0] += trade

    portvals = holdings[symbol] * prices[symbol] + cash
    return pd.DataFrame(portvals, columns=["Portfolio Value"])
