import pandas as pd
import numpy as np
from util import get_data

def compute_portvals(trades_df, start_val=100000, commission=9.95, impact=0.005):
    """
    Compute portfolio values.

    Parameters:
    trades_df: DataFrame with one column representing the stock symbol,
               and values representing daily trades (+1000, -1000, +2000, -2000, 0).
               Index is dates.
    start_val: Initial cash value
    commission: Commission cost per transaction
    impact: Market impact

    Returns:
    DataFrame with portfolio value for each trading day
    """
    # Extract symbol from the column name
    symbols = trades_df.columns.tolist()
    # Assuming only one symbol is traded in this trades_df as per project structure
    if len(symbols) != 1:
        raise ValueError("trades_df is expected to have exactly one stock symbol column.")
    symbol = symbols[0]

    dates = trades_df.index # Use dates from the trades_df
    start_date = dates.min()
    end_date = dates.max()

    # Get price data for the symbol
    prices_all = get_data([symbol], pd.date_range(start_date, end_date))
    prices = prices_all[[symbol]].ffill().bfill() # Ensure prices match trades_df index and fill NaNs
    prices['CASH'] = 1.0 # Add cash column

    # Reindex trades_df to ensure it aligns with prices
    full_dates = prices.index
    trades = pd.DataFrame(0, index=full_dates, columns=[symbol, 'CASH']) # Initialize with symbol and CASH column
    
    # Populate trades DataFrame based on input trades_df
    # Only process non-zero trades
    for date, trade_val in trades_df[symbol].items():
        if trade_val != 0:
            current_price = prices.loc[date, symbol]
            
            # Calculate actual shares to buy/sell based on the trade_val (e.g., -2000 means sell 2000)
            shares_to_trade = abs(trade_val)
            
            if trade_val > 0: # BUY action (+1000, +2000)
                cost = current_price * shares_to_trade * (1 + impact)
                trades.loc[date, symbol] += shares_to_trade
                trades.loc[date, 'CASH'] -= (cost + commission)
            elif trade_val < 0: # SELL action (-1000, -2000)
                proceeds = current_price * shares_to_trade * (1 - impact)
                trades.loc[date, symbol] -= shares_to_trade
                trades.loc[date, 'CASH'] += (proceeds - commission)

    # Holdings: cumulative sum
    holdings = trades.cumsum()

    # Origin
    # holdings.loc[holdings.index[0], 'CASH'] += start_val # Add starting cash to the first day
    # Change
    holdings.iloc[0, holdings.columns.get_loc('CASH')] = start_val + trades.iloc[0]['CASH']

    # Daily portfolio value
    # Make sure prices and holdings have the same columns for multiplication
    portvals = (holdings * prices.loc[holdings.index, holdings.columns]).sum(axis=1)
    
    return portvals.to_frame(name='Value')

