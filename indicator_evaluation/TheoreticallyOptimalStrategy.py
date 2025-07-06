'''
Allowable positions are 1000 shares long, 1000 shares short, 0 shares. (You may trade up to 2000 shares at a time as long as your positions are 1000 shares long or 1000 shares short.)  
Benchmark: The performance of a portfolio starting with $100,000 cash, investing in 1000 shares of JPM, and holding that position.  
Transaction costs for TheoreticallyOptimalStrategy:  
Commission: $0.00
Impact: 0.00. 
'''

import pandas as pd
import datetime as dt
from util import get_data

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

def testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
    # Get stock prices for the given dates
    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates, addSPY=False).dropna()

    df_trades = pd.DataFrame(data=0, index=prices.index, columns=[symbol])

    current_holdings = 0

    for i in range(prices.shape[0] - 1):
        current_date = prices.index[i]
        next_date = prices.index[i+1]

        current_price = prices.loc[current_date, symbol]
        next_price = prices.loc[next_date, symbol]

        if next_price > current_price:
            if current_holdings < 1000:
                shares_to_buy = 1000 - current_holdings
                df_trades.loc[current_date, symbol] += shares_to_buy
                current_holdings += shares_to_buy
        elif next_price < current_price:
            if current_holdings > -1000:
                shares_to_sell = -1000 - current_holdings
                df_trades.loc[current_date, symbol] += shares_to_sell
                current_holdings += shares_to_sell
        else:
            df_trades.loc[current_date, symbol] += 0

    if current_holdings > 0:
        df_trades.loc[prices.index[-1], symbol] -= current_holdings
    elif current_holdings < 0:
        df_trades.loc[prices.index[-1], symbol] -= current_holdings

    return df_trades