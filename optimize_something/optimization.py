""""""  		  	   		 	 	 			  		 			 	 	 		 		 	
"""MC1-P2: Optimize a portfolio.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
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
import numpy as np  		  	   		 	 	 			  		 			 	 	 		 		 	
import matplotlib.pyplot as plt  		  	   		 	 	 			  		 			 	 	 		 		 	
import pandas as pd  		  	   		 	 	 			  		 			 	 	 		 		 	
from util import get_data, plot_data  
import scipy.optimize as spo		  	   		 	 	 			  		 			 	 	 		 		 	

def author():  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    :return: The GT username of the student  		  	   		 	 	 			  		 			 	 	 		 		 	
    :rtype: str  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    return "awang758"  # replace tb34 with your Georgia Tech username.
  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
def gtid():  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    :return: The GT ID of the student  		  	   		 	 	 			  		 			 	 	 		 		 	
    :rtype: int  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    return 904081341  # replace with your GT ID number
  		  	   		 	 	 			  		 			 	 	 		 		 	
def study_group():
    return "awang758"  		  	   		 	 	 			  		 			 	 	 		 		 	
# This is the function that will be tested by the autograder  		  	   		 	 	 			  		 			 	 	 		 		 	
# The student must update this code to properly implement the functionality  	
# #################################################################################### #	  	   		 	 	 			  		 			 	 	 		 		 	
def normalize_prices(prices):
    return prices / prices.iloc[0]

# Step 2: Compute daily returns
def compute_daily_returns(prices):
    return prices.pct_change().fillna(0)

# Step 3: Portfolio value calculation
def compute_portfolio_value(normalized_prices, allocations, start_val=1_000_000):
    alloced = normalized_prices * allocations
    position_values = alloced * start_val
    portfolio_value = position_values.sum(axis=1)
    return portfolio_value

# Step 4: Objective function (negative Sharpe Ratio)
def negative_sharpe_ratio(allocs, normalized_prices, rfr=0.0, sf=252.0):
    portfolio_value = compute_portfolio_value(normalized_prices, allocs)
    daily_returns = compute_daily_returns(portfolio_value)
    excess_daily_returns = daily_returns - rfr / sf
    sharpe_ratio = (excess_daily_returns.mean() / excess_daily_returns.std()) * np.sqrt(sf)
    return -sharpe_ratio  # negate for minimization

def optimize_portfolio(  		  	   		 	 	 			  		 			 	 	 		 		 	
    sd=dt.datetime(2008, 1, 1),  		  	   		 	 	 			  		 			 	 	 		 		 	
    ed=dt.datetime(2009, 1, 1),  		  	   		 	 	 			  		 			 	 	 		 		 	
    syms=["GOOG", "AAPL", "GLD", "XOM"],  		  	   		 	 	 			  		 			 	 	 		 		 	
    gen_plot=False,  		  	   		 	 	 			  		 			 	 	 		 		 	
):  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		 	 	 			  		 			 	 	 		 		 	
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		 	 	 			  		 			 	 	 		 		 	
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		 	 	 			  		 			 	 	 		 		 	
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		 	 	 			  		 			 	 	 		 		 	
    statistics.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type sd: datetime  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type ed: datetime  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		 	 	 			  		 			 	 	 		 		 	
        symbol in the data directory)  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type syms: list  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		 	 	 			  		 			 	 	 		 		 	
        code with gen_plot = False.  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type gen_plot: bool  		  	   		 	 	 			  		 			 	 	 		 		 	
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		 	 	 			  		 			 	 	 		 		 	
        standard deviation of daily returns, and Sharpe ratio  		  	   		 	 	 			  		 			 	 	 		 		 	
    :rtype: tuple  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Read in adjusted closing prices for given symbols, date range  		  	   		 	 	 			  		 			 	 	 		 		 	
    dates = pd.date_range(sd, ed)  		  	   		 	 	 			  		 			 	 	 		 		 	
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices_all.ffill(inplace=True)
    prices_all.bfill(inplace=True)


    prices = prices_all[syms]  # only portfolio symbols  		  	   		 	 	 			  		 			 	 	 		 		 	
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later 
    normalized_prices = normalize_prices(prices)
    normalized_SPY = normalize_prices(prices_SPY) 		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # find the allocations for the optimal portfolio  		  	   		 	 	 			  		 			 	 	 		 		 	
    # note that the values here ARE NOT meant to be correct for a test case  		  	   		 	 	 			  		 			 	 	 		 		 	
    # allocs = np.asarray(  		  	   		 	 	 			  		 			 	 	 		 		 	
    #     [0.2, 0.2, 0.3, 0.3]  		  	   		 	 	 			  		 			 	 	 		 		 	
    # )  # add code here to find the allocations  		  	   		 	 	 			  		 			 	 	 		 		 	
    # cr, adr, sddr, sr = [  		  	   		 	 	 			  		 			 	 	 		 		 	
    #     0.25,  		  	   		 	 	 			  		 			 	 	 		 		 	
    #     0.001,  		  	   		 	 	 			  		 			 	 	 		 		 	
    #     0.0005,  		  	   		 	 	 			  		 			 	 	 		 		 	
    #     2.1,  		  	   		 	 	 			  		 			 	 	 		 		 	
    # ]  # add code here to compute stats  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # # Get daily portfolio value  		  	   		 	 	 			  		 			 	 	 		 		 	
    # port_val = prices_SPY  # add code here to compute daily portfolio values  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # # Compare daily portfolio value with SPY using a normalized plot  		  	   		 	 	 			  		 			 	 	 		 		 	
    # if gen_plot:  		  	   		 	 	 			  		 			 	 	 		 		 	
    #     # add code to plot here  		  	   		 	 	 			  		 			 	 	 		 		 	
    #     df_temp = pd.concat(  		  	   		 	 	 			  		 			 	 	 		 		 	
    #         [port_val, prices_SPY], keys=["Portfolio", "SPY"], axis=1  		  	   		 	 	 			  		 			 	 	 		 		 	
    #     )  		  	   		 	 	 			  		 			 	 	 		 		 	
    #     pass  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # return allocs, cr, adr, sddr, sr  
    num_assets = len(syms)
    initial_guess = np.ones(num_assets) / num_assets
    bounds = [(0.0, 1.0) for _ in range(num_assets)]
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    result = spo.minimize(negative_sharpe_ratio, initial_guess,
                          args=(normalized_prices,),
                          method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'disp': False})

    allocs = result.x
    port_val = compute_portfolio_value(normalized_prices, allocs)
    daily_returns = compute_daily_returns(port_val)

    # Step 5.3: Performance metrics
    cr = (port_val[-1] / port_val[0]) - 1
    adr = daily_returns.mean()
    sddr = daily_returns.std()
    sr = (adr / sddr) * np.sqrt(252)

    # Step 5.4: Plotting
    if gen_plot:
        df_temp = pd.concat([port_val / port_val.iloc[0],
                             normalized_SPY], axis=1)
        df_temp.columns = ['Portfolio', 'SPY']
        df_temp.plot(title="Daily Portfolio Value vs. SPY", fontsize=10)
        plt.xlabel("Date")
        plt.ylabel("Normalized Price")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('Figure1.png')
        plt.show()

    return allocs, cr, adr, sddr, sr		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
def test_code():  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    This function WILL NOT be called by the auto grader.  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    start_date = dt.datetime(2009, 1, 1)  		  	   		 	 	 			  		 			 	 	 		 		 	
    end_date = dt.datetime(2010, 1, 1)  		  	   		 	 	 			  		 			 	 	 		 		 	
    symbols = ["GOOG", "AAPL", "GLD", "XOM", "IBM"]  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Assess the portfolio  		  	   		 	 	 			  		 			 	 	 		 		 	
    allocations, cr, adr, sddr, sr = optimize_portfolio(  		  	   		 	 	 			  		 			 	 	 		 		 	
        sd=start_date, ed=end_date, syms=symbols, gen_plot=False  		  	   		 	 	 			  		 			 	 	 		 		 	
    )  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Print statistics  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Start Date: {start_date}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"End Date: {end_date}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Symbols: {symbols}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Allocations:{allocations}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Sharpe Ratio: {sr}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Average Daily Return: {adr}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Cumulative Return: {cr}")  		  	   		 	 	 			  		 			 	 	 		 		 	


    # Plotting the optimized portfolio vs SPY
    prices_all = get_data(symbols + ["SPY"], pd.date_range(start_date, end_date))
    prices = prices_all[symbols]
    prices_SPY = prices_all["SPY"]

    normed_prices = normalize_prices(prices)
    port_val = compute_portfolio_value(normed_prices, allocations)

    # Normalize SPY for comparison
    normed_SPY = normalize_prices(prices_SPY)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(port_val, label='Optimized Portfolio')
    plt.plot(normed_SPY, label='SPY')
    plt.title("Optimized Portfolio vs SPY")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot.png")  # Save as required
    plt.show()	  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
if __name__ == "__main__":  		  	   		 	 	 			  		 			 	 	 		 		 	
    # This code WILL NOT be called by the auto grader  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Do not assume that it will be called  		  	   		 	 	 			  		 			 	 	 		 		 	
    test_code()  		  	   		 	 	 			  		 			 	 	 		 		 	
