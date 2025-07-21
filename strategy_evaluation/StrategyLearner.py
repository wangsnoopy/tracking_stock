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
import numpy as np
import indicators
import QLearner as ql

class StrategyLearner:
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.learner = ql.QLearner(
            num_states=1000,
            num_actions=3,
            alpha=0.2,
            gamma=0.9,
            rar=0.6,
            radr=0.95,
            dyna=50,
            verbose=False
        )
        self.sma_window = 20
        self.bb_window = 20
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9

    def author(self):
        return "awang758"
    # This is the origin add_evidence
    def add_evidence(self, symbol="IBM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), sv=100000):
        self.sv = sv
        self.symbol = symbol
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)
        prices = prices_all[[symbol]].ffill().bfill()

        # Compute indicators
        bbp = indicators.bollinger_bands_percentage(prices)
        macd = indicators.macd_histogram(prices)
        rsi = indicators.rsi(prices)

        # Normalize
        bbp = (bbp - bbp.mean()) / bbp.std()
        macd = (macd - macd.mean()) / macd.std()
        rsi = (rsi - rsi.mean()) / rsi.std()

        # Combine
        df_indicators = pd.concat([bbp, macd, rsi], axis=1).dropna()
        df_indicators.columns = ['BBP', 'MACD', 'RSI']

        # Future return
        returns = prices.shift(-5) / prices - 1.0
        returns = returns[symbol].loc[df_indicators.index]

        # Discretize
        states = self._compute_states(df_indicators)

        # origin lopp
        for epoch in range(20):
            for i in range(len(states) - 5):
                s = states[i]
                r = returns.iloc[i + 5] * 1000  # reward scaling
                if i == 0:
                    self.learner.querysetstate(s)
                else:
                    self.learner.query(s, r)

    # Origin 
    def testPolicy(self, symbol="IBM", sd=dt.datetime(2009,1,1), ed=dt.datetime(2010,1,1), sv=100000):
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)
        prices = prices_all[[symbol]].ffill().bfill()

        # Compute indicators
        bbp = indicators.bollinger_bands_percentage(prices)
        macd = indicators.macd_histogram(prices)
        rsi = indicators.rsi(prices)

        # Normalize
        bbp = (bbp - bbp.mean()) / bbp.std()
        macd = (macd - macd.mean()) / macd.std()
        rsi = (rsi - rsi.mean()) / rsi.std()

        df_indicators = pd.concat([bbp, macd, rsi], axis=1).dropna()
        df_indicators.columns = ['BBP', 'MACD', 'RSI']

        trades = pd.DataFrame(0, index=prices.index, columns=[symbol])
        holdings = 0

        states = self._compute_states(df_indicators)

        for i, date in enumerate(df_indicators.index):
            s = states[i]
            action = self.learner.querysetstate(s)

            if action == 0:  # sell
                if holdings == 1000:
                    trades.at[date, symbol] = -2000
                    holdings = -1000
                elif holdings == 0:
                    trades.at[date, symbol] = -1000
                    holdings = -1000
            elif action == 2:  # buy
                if holdings == -1000:
                    trades.at[date, symbol] = 2000
                    holdings = 1000
                elif holdings == 0:
                    trades.at[date, symbol] = 1000
                    holdings = 1000
            # action == 1 → hold

        return trades

    def _compute_states(self, df):
        # Discretize each indicator into 10 bins and create composite state
        bins = 10
        bbp_bins = pd.qcut(df['BBP'], bins, labels=False, duplicates='drop')
        macd_bins = pd.qcut(df['MACD'], bins, labels=False, duplicates='drop')
        rsi_bins = pd.qcut(df['RSI'], bins, labels=False, duplicates='drop')

        states = (bbp_bins * 100) + (macd_bins * 10) + rsi_bins
        return states.fillna(0).astype(int).values