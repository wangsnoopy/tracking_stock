""""""  		  	   		 	 	 			  		 			 	 	 		 		 	
"""MC2-P1: Market simulator.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			 	 	 		 		 	
or edited.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Student Name: Tucker Balch (replace with your name)  		  	   		 	 	 			  		 			 	 	 		 		 	
GT User ID: tb34 (replace with your User ID)  		  	   		 	 	 			  		 			 	 	 		 		 	
GT ID: 900897987 (replace with your GT ID)  		  	   		 	 	 			  		 			 	 	 		 		 	
"""  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
import datetime as dt  		  	   		 	 	 			  		 			 	 	 		 		 	
import os  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
import numpy as np  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
import pandas as pd  		  	   		 	 	 			  		 			 	 	 		 		 	
from util import get_data, plot_data  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
def author():  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    :return: The GT username of the student  		  	   		 	 	 			  		 			 	 	 		 		 	
    :rtype: str  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    return "awang758"
  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
def gtid():  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    :return: The GT ID of the student  		  	   		 	 	 			  		 			 	 	 		 		 	
    :rtype: int  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    return 904081341
  		  	   		 	 	 			  		 			 	 	 		 		 	
def study_group():
    return "awang758"
	  	   		 	 	 			  		 			 	 	 		 		 	
def compute_portvals(
    orders_file="./orders/orders.csv",
    start_val=1000000,
    commission=9.95,
    impact=0.005,
):
    """
    Computes the portfolio values.

    :param orders_file: Path of the order file or the file object
    :type orders_file: str or file object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """

    # Read orders file and sort by date
    orders_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    orders_df = orders_df.sort_index()

    # Determine the date range for the simulation
    start_date = orders_df.index.min()
    end_date = orders_df.index.max()

    # Get unique symbols from orders
    symbols = orders_df['Symbol'].unique().tolist()

    # Get adjusted close prices for all symbols within the date range
    # Ensure SPY is included if you plan to compare against it later (though not for compute_portvals return)
    prices_df = get_data(symbols, pd.date_range(start_date, end_date), addSPY=False)
    
    # Fill missing price data (e.g., weekends, holidays)
    # Forward fill then backward fill to handle NaNs at the beginning
    prices_df.fillna(method='ffill', inplace=True)
    prices_df.fillna(method='bfill', inplace=True)

    # Initialize trades dataframe to record daily changes in holdings and cash
    # This dataframe will store the net effect of transactions on each day
    trades_df = prices_df.copy() # Use prices_df index and columns for alignment
    trades_df[:] = 0.0 # Initialize with zeros
    trades_df['cash'] = 0.0 # Add a column for cash movements

    # Process each order
    for index, row in orders_df.iterrows():
        trade_date = index
        symbol = row['Symbol']
        order_type = row['Order']
        shares = row['Shares']

        # Get the stock price on the trade date
        price = prices_df.loc[trade_date, symbol]

        # Calculate cost/revenue and apply transaction costs
        if order_type == 'BUY':
            actual_price = price * (1 + impact) # Price moves against buyer
            cost = shares * actual_price
            trades_df.loc[trade_date, symbol] += shares
            trades_df.loc[trade_date, 'cash'] -= (cost + commission)
        elif order_type == 'SELL':
            actual_price = price * (1 - impact) # Price moves against seller
            revenue = shares * actual_price
            trades_df.loc[trade_date, symbol] -= shares
            trades_df.loc[trade_date, 'cash'] += (revenue - commission)

    # Initialize holdings and cash based on trades
    # The initial cash value is start_val
    holdings_df = prices_df.copy()
    holdings_df[:] = 0 # Holdings start at zero for all stocks

    # Apply the trades to get cumulative holdings over time
    for col in symbols: # Iterate over stock columns
        holdings_df[col] = trades_df[col].cumsum() # Cumulative sum of shares traded

    # Calculate cumulative cash balance
    # Initial cash + cumulative cash movements
    holdings_df['cash'] = start_val + trades_df['cash'].cumsum()

    # Calculate portfolio value for each day
    # Value = (shares * price for each stock) + cash
    portvals = (holdings_df[symbols] * prices_df[symbols]).sum(axis=1) + holdings_df['cash']

    # Convert to DataFrame with a single column (column name doesn't matter)
    portvals_df = pd.DataFrame(portvals, columns=['Portfolio_Value'])

    return portvals_df  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
def get_portfolio_stats(portvals):
    daily_returns = portvals.pct_change().dropna()
    cr = (portvals.iloc[-1] / portvals.iloc[0]) - 1
    adr = daily_returns.mean()[0]
    sddr = daily_returns.std()[0]
    sr = (adr / sddr) * np.sqrt(252)
    return cr[0], adr, sddr, sr


def test_code():  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    Helper function to test code  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    # this is a helper function you can use to test your code  		  	   		 	 	 			  		 			 	 	 		 		 	
    # note that during autograding his function will not be called.  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Define input parameters  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    of = "./orders/orders-10.csv"
    sv = 1000000  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Process orders  		  	   		 	 	 			  		 			 	 	 		 		 	
    portvals = compute_portvals(orders_file=of, start_val=sv)  		  	   		 	 	 			  		 			 	 	 		 		 	
    if isinstance(portvals, pd.DataFrame):  		  	   		 	 	 			  		 			 	 	 		 		 	
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		 	 	 			  		 			 	 	 		 		 	
    else:  		  	   		 	 	 			  		 			 	 	 		 		 	
        "warning, code did not return a DataFrame" 

    cr, adr, sddr, sr = get_portfolio_stats(portvals) 		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # SPY benchmark
    # start_date = portvals.index.min()
    # end_date = portvals.index.max()
    start_date = dt.datetime(2011, 1, 10)
    end_date = dt.datetime(2011, 8, 1)
    prices_SPY = get_data(['SPY'], pd.date_range(start_date, end_date))
    prices_SPY = prices_SPY.fillna(method="ffill").fillna(method="bfill")
    norm_SPY = prices_SPY['SPY'] / prices_SPY['SPY'].iloc[0]
    daily_returns_SPY = norm_SPY.pct_change().dropna()
    cr_SPY = (norm_SPY.iloc[-1] / norm_SPY.iloc[0]) - 1
    adr_SPY = daily_returns_SPY.mean()
    sddr_SPY = daily_returns_SPY.std()
    sr_SPY = (adr_SPY / sddr_SPY) * np.sqrt(252)
 		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Compare portfolio against $SPX                                                                                        
    print(f"Date Range: {start_date.date()} to {end_date.date()}")
    print(f"Sharpe Ratio of Fund: {sr:.4f}")
    print(f"Sharpe Ratio of SPY : {sr_SPY:.4f}")
    print(f"Cumulative Return of Fund: {cr:.4f}")
    print(f"Cumulative Return of SPY : {cr_SPY:.4f}")
    print(f"Standard Deviation of Fund: {sddr:.4f}")
    print(f"Standard Deviation of SPY : {sddr_SPY:.4f}")
    print(f"Average Daily Return of Fund: {adr:.4f}")
    print(f"Average Daily Return of SPY : {adr_SPY:.4f}")
    print(f"Final Portfolio Value: {portvals.iloc[-1, 0]:.2f}")  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
if __name__ == "__main__":  		  	   		 	 	 			  		 			 	 	 		 		 	
    test_code()  		  	   		 	 	 			  		 			 	 	 		 		 	
