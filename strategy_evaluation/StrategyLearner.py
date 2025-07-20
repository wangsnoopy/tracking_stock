""""""  		  	   		 	 	 			  		 			 	 	 		 		 	
"""  		  	   		 	 	 			  		 			 	 	 		 		 	
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
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
import pandas as pd
import util as ut
import random
import numpy as np
import indicators
import QLearner as ql

class StrategyLearner:
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.learner = ql.QLearner(num_states=1000, num_actions=3, alpha=0.2,
                                   gamma=0.9, rar=0.5, radr=0.99, dyna=0, verbose=False)

    def add_evidence(self, symbol="IBM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), sv=100000):
        self.sv = sv
        prices_all = ut.get_data([symbol], pd.date_range(sd, ed))
        prices = prices_all[[symbol]].fillna(method="ffill").fillna(method="bfill")

        # Calculate indicators
        bb = indicators.BB(prices)
        macd = indicators.MACD(prices)
        rsi = indicators.RSI(prices)

        # Normalize indicators
        bb = (bb - bb.mean()) / bb.std()
        macd = (macd - macd.mean()) / macd.std()
        rsi = (rsi - rsi.mean()) / rsi.std()

        # Combine into one DataFrame
        indicators_df = pd.concat([bb, macd, rsi], axis=1).dropna()
        indicators_df.columns = ['BB', 'MACD', 'RSI']

        # Discretize states
        states = self._compute_states(indicators_df)

        # Calculate returns
        future_prices = prices.shift(-5)
        returns = (future_prices / prices) - 1.0

        df = pd.DataFrame(index=indicators_df.index)
        df['prices'] = prices[symbol]
        df['returns'] = returns[symbol]

        # Train learner
        for i in range(10):  # Multiple passes for convergence
            for j in range(len(df.index) - 5):
                date = df.index[j]
                next_date = df.index[j + 1]

                s = states[j]
                r = df['returns'].iloc[j + 5] * 1000  # Scale reward
                a = self.learner.query(s, r)

    def testPolicy(self, symbol="IBM", sd=dt.datetime(2009,1,1), ed=dt.datetime(2010,1,1), sv=100000):
        prices_all = ut.get_data([symbol], pd.date_range(sd, ed))
        prices = prices_all[[symbol]].fillna(method="ffill").fillna(method="bfill")

        # Compute indicators
        bb = indicators.BB(prices)
        macd = indicators.MACD(prices)
        rsi = indicators.RSI(prices)

        bb = (bb - bb.mean()) / bb.std()
        macd = (macd - macd.mean()) / macd.std()
        rsi = (rsi - rsi.mean()) / rsi.std()

        indicators_df = pd.concat([bb, macd, rsi], axis=1).dropna()
        indicators_df.columns = ['BB', 'MACD', 'RSI']

        trades = pd.DataFrame(index=prices.index)
        trades[symbol] = 0
        holdings = 0

        states = self._compute_states(indicators_df)

        for i in range(len(states)):
            s = states[i]
            a = self.learner.querysetstate(s)

            if a == 0:  # Sell
                if holdings == 1000:
                    trades.iloc[i][symbol] = -2000
                    holdings = -1000
                elif holdings == 0:
                    trades.iloc[i][symbol] = -1000
                    holdings = -1000
            elif a == 2:  # Buy
                if holdings == -1000:
                    trades.iloc[i][symbol] = 2000
                    holdings = 1000
                elif holdings == 0:
                    trades.iloc[i][symbol] = 1000
                    holdings = 1000
            # Action 1 is hold → no trade

        return trades

    def _compute_states(self, df):
        bins = 10
        bb_bins = pd.qcut(df['BB'], bins, labels=False, duplicates='drop')
        macd_bins = pd.qcut(df['MACD'], bins, labels=False, duplicates='drop')
        rsi_bins = pd.qcut(df['RSI'], bins, labels=False, duplicates='drop')

        states = (bb_bins * 100) + (macd_bins * 10) + rsi_bins
        return states.fillna(0).astype(int).values
