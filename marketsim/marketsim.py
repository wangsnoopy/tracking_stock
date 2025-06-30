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
    # this is the function the autograder will call to test your code  		  	   		   	 		  		  		    	 		 		   		 		  
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		   	 		  		  		    	 		 		   		 		  
    # code should work correctly with either input

    # --------------- Orders Data --------------- #
    # Organize data into df
    # Read and organize orders data
    orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])

    # Get start and end dates
    start_date = orders.index.min()
    end_date = orders.index.max()

    # Get list of unique stocks
    unique_stocks = list(orders['Symbol'].unique())

    # Get stock prices for the date range
    prices = get_data(unique_stocks, pd.date_range(start_date, end_date)).drop(columns=["SPY"], errors='ignore')
    prices["Cash"] = 1.0  # Cash column with ones

    # Initialize trades DataFrame
    trades = pd.DataFrame(0.0, columns=prices.columns, index=prices.index)

    # Process orders to populate trades
    for date, order in orders.iterrows():
        symbol = order['Symbol']
        shares = order['Shares']
        position_factor = 1 if order['Order'] == "BUY" else -1

        # Skip specific date for bonus points (if applicable)
        if date.date() == dt.date(2011, 6, 15):
            continue

        # Update trades for the stock
        trades.at[date, symbol] += shares * position_factor

        # Calculate cash impact including commission and market impact
        price = prices.at[date, symbol]
        cash_impact = (-position_factor) * price * shares
        market_impact_fee = impact * abs(cash_impact)
        trades.at[date, "Cash"] += cash_impact - market_impact_fee - commission

    # Initialize holdings DataFrame
    holdings = pd.DataFrame(0.0, columns=trades.columns, index=trades.index)
    holdings.iloc[0] = trades.iloc[0]
    holdings.at[holdings.index[0], "Cash"] += float(start_val)

    # Compute holdings for each day
    for i in range(1, len(holdings)):
        holdings.iloc[i] = holdings.iloc[i-1] + trades.iloc[i]

    # Calculate portfolio values
    values = prices * holdings
    port_vals = values.sum(axis=1)

    # Return as a single-column DataFrame
    return pd.DataFrame(port_vals, columns=['Portfolio Value']) 		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 	
def test_code():  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    Helper function to test code  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    # this is a helper function you can use to test your code  		  	   		 	 	 			  		 			 	 	 		 		 	
    # note that during autograding his function will not be called.  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Define input parameters  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    of = "./orders/orders-10.csv"
    sv = 1000000  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Compute portfolio values
    portvals = compute_portvals(orders_file=of, start_val=sv, commission=9.95, impact=0.005)

    # Calculate Fund metrics
    daily_returns = portvals['Portfolio Value'].pct_change().dropna()
    cumulative_return = (portvals['Portfolio Value'].iloc[-1] / portvals['Portfolio Value'].iloc[0]) - 1
    average_daily_return = daily_returns.mean()
    std_daily_returns = daily_returns.std(ddof=1)
    risk_free_rate = 0
    sharpe_ratio = np.sqrt(252) * (average_daily_return - risk_free_rate) / std_daily_returns

    # SPY benchmark
    start_date = portvals.index.min()
    end_date = portvals.index.max()
    # Get SPY prices for the same date range and align with portvals
    spy_prices = get_data(['SPY'], pd.date_range(start_date, end_date), addSPY=True)
    spy_prices.fillna(method='ffill', inplace=True)
    spy_prices.fillna(method='bfill', inplace=True)

    # Align SPY with portfolio dates exactly
    spy_prices = spy_prices.reindex(portvals.index)  # <--- THIS is the key fix

    spy_daily_returns = spy_prices['SPY'].pct_change().dropna()
    cumulative_return_spy = (spy_prices['SPY'].iloc[-1] / spy_prices['SPY'].iloc[0]) - 1
    average_daily_return_spy = spy_daily_returns.mean()
    std_daily_returns_spy = spy_daily_returns.std(ddof=1)
    sharpe_ratio_spy = np.sqrt(252) * (average_daily_return_spy / std_daily_returns_spy)

    # Print results
    print(f"Date Range: {start_date} to {end_date}")
    print("-" * 30)
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of $SPX: {sharpe_ratio_spy}")
    print()
    print(f"Cumulative Return of Fund: {cumulative_return}")
    print(f"Cumulative Return of $SPX: {cumulative_return_spy}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_returns}")
    print(f"Standard Deviation of $SPX: {std_daily_returns_spy}")
    print()
    print(f"Average Daily Return of Fund: {average_daily_return}")
    print(f"Average Daily Return of $SPX: {average_daily_return_spy}")
    print()
    print(f"Final Portfolio Value: {portvals['Portfolio Value'].iloc[-1]}") 		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
if __name__ == "__main__":  		  	   		 	 	 			  		 			 	 	 		 		 	
    test_code()  		  	   		 	 	 			  		 			 	 	 		 		 	
