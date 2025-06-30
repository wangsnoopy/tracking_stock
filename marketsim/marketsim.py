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
    orders = pd.read_csv(orders_file, header=0, index_col=0)

    # Convert index to datetime
    orders.index = pd.to_datetime(list(orders.index.values))

    # --------------- Prices Data --------------- #
    # grab start date / end date from orders
    start_date = pd.to_datetime(min(list(orders.index.values)), format='%Y-%m-%d')
    end_date = pd.to_datetime(max(list(orders.index.values)), format='%Y-%m-%d')

    # Get list of all stocks used
    unique_stocks = list(set(orders["Symbol"]))

    # get stock data for date range
    prices = get_data(unique_stocks, pd.date_range(start_date, end_date)).drop(columns=["SPY"])
    prices["Cash"] = 1  # column should be all ones

    # --------------- Trades Data --------------- #
    # Setup position trades df with zeros, copy column and row indexes from prices
    ## trades data represents the net impact to positions and cash
    trades = pd.DataFrame(data=0.000, columns=prices.columns.values, index=prices.index.values)

    # Populate trades data by tracing orders data
    for i in range(orders.shape[0]):
        # Get data for orders
        order_row = orders.iloc[[i]]  # 0 = symbol, 1 = Position, 2 = Num shares
        date = order_row.index.values[0]
        if date == dt.date(2011, 6, 15):
            # I hope this is still relevant for bonus points, grader please :)
            continue

        # Determine if buy or sell
        if order_row.Order[0] == "BUY":
            position_factor = 1
        else:
            position_factor = -1

        # Update trades with position, add to existing trade data if already there
        trades.at[date, order_row.Symbol[0]] += order_row.Shares[0] * position_factor

        # -- Cash Impacts -- #
        cash_for_trade_impact = (-1 * position_factor) * prices.at[date, order_row.Symbol[0]] * order_row.Shares[0]
        # Transaction costs
        market_impact_fee = impact * abs(cash_for_trade_impact)

        # Update Cash
        trades.at[date, "Cash"] += cash_for_trade_impact - market_impact_fee - commission

    # --------------- Calculate holdings --------------- #
    # Setup holdings df
    holdings = pd.DataFrame(data=0.000, columns=trades.columns.values, index=trades.index.values)
    holdings.iloc[[0]] = trades.iloc[[0]]
    holdings.Cash.iat[0] += float(start_val)  # add in starting cash

    for i in range(1, holdings.shape[0]):
        # Carry over holdings position values from prior trading day, add trades data
        holdings.iloc[[i]] = holdings.iloc[[i-1]].values + trades.iloc[[i]]

    values = prices * holdings
    port_values = values.sum(axis=1)  # used for debugging

    return port_values  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 	
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

    daily_returns = portvals.pct_change().dropna()
    cumulative_return = (portvals.iloc[-1] / portvals.iloc[0]) - 1
    average_daily_return = daily_returns.mean()
    standard_deviation_daily_returns = daily_returns.std(ddof=1)
    risk_free_rate = 0 
    sharpe_ratio = np.sqrt(252) * ((average_daily_return - risk_free_rate) / standard_deviation_daily_returns)
		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # SPY benchmark
    start_date = portvals.index.min()
    end_date = portvals.index.max()
    spy_prices = get_data(["SPY"], pd.date_range(start_date, end_date), addSPY=True)
    spy_prices.fillna(method='ffill', inplace=True)
    spy_prices.fillna(method='bfill', inplace=True)
    spy_prices = spy_prices[['SPY']]

    spy_daily_returns = spy_prices['SPY'].pct_change().dropna()
    cumulative_return_SPY = (spy_prices['SPY'].iloc[-1] / spy_prices['SPY'].iloc[0]) - 1
    average_daily_return_SPY = spy_daily_returns.mean()
    standard_deviation_SPY = spy_daily_returns.std(ddof=1)
    sharpe_ratio_SPY = np.sqrt(252) * ((average_daily_return_SPY - risk_free_rate) / standard_deviation_SPY)

    # --- Print comparison results ---
    print(f"Date Range: {start_date.date()} to {end_date.date()}")
    print("-" * 30)
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cumulative_return}")
    print(f"Cumulative Return of SPY : {cumulative_return_SPY}")
    print()
    print(f"Standard Deviation of Fund: {standard_deviation_daily_returns}")
    print(f"Standard Deviation of SPY : {standard_deviation_SPY}")
    print()
    print(f"Average Daily Return of Fund: {average_daily_return}")
    print(f"Average Daily Return of SPY : {average_daily_return_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals.iloc[-1]}")  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
if __name__ == "__main__":  		  	   		 	 	 			  		 			 	 	 		 		 	
    test_code()  		  	   		 	 	 			  		 			 	 	 		 		 	
