import pandas as pd
import numpy as np
from util import get_data

def compute_portvals(orders_df, start_val=1000000, commission=9.95, impact=0.005):
    """
    Compute portfolio values.
    
    Parameters:
    orders_df: DataFrame with columns ['Symbol', 'Order', 'Shares'] and index as dates
    start_val: Initial cash value
    commission: Commission cost per transaction
    impact: Market impact
    
    Returns:
    DataFrame with portfolio value for each trading day
    """
    # Extract trading dates
    dates = orders_df.index
    symbols = orders_df['Symbol'].unique().tolist()

    # Get price data for all symbols
    prices = get_data(symbols, dates.min(), dates.max())
    prices = prices[symbols]
    prices['CASH'] = 1.0

    # Initialize trades and portfolio value DataFrames
    trades = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    
    for date, row in orders_df.iterrows():
        symbol = row['Symbol']
        order = row['Order']
        shares = row['Shares']
        price = prices.loc[date, symbol]
        
        # Market impact adjustment
        if order == 'BUY':
            cost = price * shares * (1 + impact)
            trades.loc[date, symbol] += shares
            trades.loc[date, 'CASH'] -= (cost + commission)
        elif order == 'SELL':
            proceeds = price * shares * (1 - impact)
            trades.loc[date, symbol] -= shares
            trades.loc[date, 'CASH'] += (proceeds - commission)

    # Holdings: cumulative sum
    holdings = trades.cumsum()
    holdings.iloc[0, holdings.columns.get_loc('CASH')] += start_val

    # Daily portfolio value
    portvals = (holdings * prices).sum(axis=1)
    return portvals.to_frame(name='Value')

