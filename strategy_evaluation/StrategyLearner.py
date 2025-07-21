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
            rar=0.5,
            radr=0.99,
            dyna=0,
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
    # def add_evidence(self, symbol="IBM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), sv=100000):
    #     self.sv = sv
    #     self.symbol = symbol
    #     dates = pd.date_range(sd, ed)
    #     prices_all = ut.get_data([symbol], dates)
    #     prices = prices_all[[symbol]].ffill().bfill()

    #     # Compute indicators
    #     bbp = indicators.bollinger_bands_percentage(prices)
    #     macd = indicators.macd_histogram(prices)
    #     rsi = indicators.rsi(prices)

    #     # Normalize
    #     bbp = (bbp - bbp.mean()) / bbp.std()
    #     macd = (macd - macd.mean()) / macd.std()
    #     rsi = (rsi - rsi.mean()) / rsi.std()

    #     # Combine
    #     df_indicators = pd.concat([bbp, macd, rsi], axis=1).dropna()
    #     df_indicators.columns = ['BBP', 'MACD', 'RSI']

    #     # Future return
    #     returns = prices.shift(-5) / prices - 1.0
    #     returns = returns[symbol].loc[df_indicators.index]

    #     # Discretize
    #     states = self._compute_states(df_indicators)

    #     for epoch in range(10):
    #         for i in range(len(states) - 5):
    #             s = states[i]
    #             r = returns.iloc[i + 5] * 1000  # reward scaling
    #             if i == 0:
    #                 self.learner.querysetstate(s)
    #             else:
    #                 self.learner.query(s, r)



    # This is try to optimize my strategy:
    def add_evidence(self, symbol="IBM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), sv=100000):
        self.sv = sv
        self.symbol = symbol
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)
        prices = prices_all[[symbol]].ffill().bfill()

        # --- CRITICAL CHANGE 1: Compute the SAME indicators as ManualStrategy ---
        sma_ratio = indicators.price_sma_ratio(prices, window=self.sma_window)
        bbp = indicators.bollinger_bands_percentage(prices, window=self.bb_window)
        macd_hist = indicators.macd_histogram(prices, fast_period=self.macd_fast,
                                             slow_period=self.macd_slow,
                                             signal_period=self.macd_signal)


        # Normalize the indicators (already doing this, good)
        # Need to handle cases where std is zero or NaN for normalization
        # Also, apply normalization consistently for all indicators
        sma_ratio_norm = (sma_ratio - sma_ratio.mean()) / sma_ratio.std()
        bbp_norm = (bbp - bbp.mean()) / bbp.std()
        macd_hist_norm = (macd_hist - macd_hist.mean()) / macd_hist.std()

        # Fill NaNs created by std=0 or initial rolling window for robustness
        sma_ratio_norm = sma_ratio_norm.replace([np.inf, -np.inf], np.nan).fillna(0)
        bbp_norm = bbp_norm.replace([np.inf, -np.inf], np.nan).fillna(0)
        macd_hist_norm = macd_hist_norm.replace([np.inf, -np.inf], np.nan).fillna(0)


        # --- CRITICAL CHANGE 2: Concatenate and name columns consistently ---
        df_indicators = pd.concat([sma_ratio_norm, bbp_norm, macd_hist_norm], axis=1).dropna()
        df_indicators.columns = ['SMA_Ratio', 'BBP', 'MACD_Hist'] # Consistent names for state computation

        # Calculate future returns (keep as is for now)
        # Ensure 'prices' and 'df_indicators' indices align for this calculation
        returns = prices.shift(-5) / prices - 1.0 # Look 5 days ahead
        returns = returns[symbol].loc[df_indicators.index] # Align with indicator data's index

        # Discretize states (updated in _compute_states)
        states = self._compute_states(df_indicators)

        # Q-learning loop
        # Consider a more sophisticated reward system (next step after fixing trades)
        # For now, let's keep your current reward, but be aware it might not be optimal
        # for driving profitable trades if the underlying asset is in a downtrend.
        
        # Determine actual price changes for reward calculation (optional, more robust reward)
        price_changes = prices.diff().loc[df_indicators.index] # Daily price change
        
        # Initialize holdings to calculate reward for each action
        current_holdings = 0 # Starting with no position for the learner's initial state
        
        # Loop through epochs and data points
        # For the reward, it's better to look at the immediate outcome of the action based on future price.
        # This will be more complex to integrate directly with your current `returns` array without
        # modifying the Q-Learner's `query` method to accept current price and holdings.
        # Let's keep your current reward structure for now, but acknowledge its limitations.

        # Store historical state-action-reward tuples for Dyna-Q if enabled
        # experience_tuples = [] # If dyna > 0, you'll collect these
        
        for epoch in range(self.learner.num_states * 10 if self.learner.dyna > 0 else 100): # Increased epochs for better learning
            # Reset position for each epoch (or each episode if your environment supported it)
            # For this simplified continuous environment, one long sequence per epoch is fine.
            current_holdings = 0
            
            for i in range(len(states)):
                s = states[i]
                
                # Get the action from the learner (exploration vs. exploitation)
                # The QLearner's `query` method should return the action chosen
                # and internally update the Q-table based on the reward received for the *previous* action.
                # Your current `querysetstate` for first state and `query` for subsequent implies
                # Q-learning, but the reward definition is crucial.

                # Let's align with the typical QLearner usage:
                # 1. Query for action based on current state.
                # 2. Perform action (conceptually).
                # 3. Observe next state and reward.
                # 4. Update Q-table.

                # Simplified Q-learning loop for this project's QLearner API:
                # QLearner's `query(s, r)` updates based on (previous state, previous action, current reward, current state)
                # So we need to compute the reward for the *previous* action taken from the *previous* state.
                
                if i == 0:
                    # For the very first state, there's no previous action or reward.
                    # Just query for the initial action.
                    action = self.learner.querysetstate(s)
                else:
                    # Calculate reward for the action taken at previous_state
                    # This is tricky because your reward `r` looks at price 5 days ahead.
                    # If action was taken on day `i-1`, the outcome is felt on `i+4`.
                    # Let's re-align the reward to be for the *action taken on day `i`*
                    # This means the reward for state `s` (day `i`) is based on future price change.
                    
                    # Reward: (portfolio value at t+5 - portfolio value at t) / initial_value
                    # This is overly simplified and doesn't directly reward per-action profit/loss.
                    # A better reward: change in portfolio value from taking action `A` at state `S` to next state `S'`.
                    # For example, reward is the immediate profit/loss from holding position based on next day's price.
                    
                    # Let's use the current `returns` and simplify the reward for learning based on actions.
                    # Reward = (price_at_t+5 / price_at_t) - 1.0
                    # This is for a "hold" strategy. We need to tie it to the *action* taken.
                    
                    # A more direct reward for your Q-Learner actions (BUY/SELL/HOLD):
                    # Action 0 (Sell/Short): Reward based on current_price - future_price (profit from shorting)
                    # Action 2 (Buy/Long): Reward based on future_price - current_price (profit from going long)
                    # Action 1 (Hold): Reward 0
                    
                    # This requires getting the price at current date and a future date relative to the action.
                    # Let's stick to the prompt's simplicity and refine your `r` for now.
                    # If returns.iloc[i] is positive, it's a "good" return.
                    
                    # For a Q-learner, the reward is received AFTER taking an action in a state and transitioning.
                    # So, if an action is taken at state[i], the reward is observed at state[i+1] (or after 5 days as per your returns).
                    # Your `returns.iloc[i+5]` implies a reward delayed by 5 days. This is valid but can make learning harder.
                    
                    # Let's simplify the reward for the learner for now to ensure it *trades*.
                    # A simple reward: -1 for a trade if price moves against, +1 if price moves with, 0 for hold.
                    # This needs to be tied to the specific action.
                    # For now, let's keep your existing reward logic as is, but be aware it's a likely point of failure for *good* performance.
                    
                    # Ensure reward `r` is for the action taken AT state `s` (which is `states[i]`)
                    # Your current reward `r = returns.iloc[i + 5] * 1000` is tied to a future return.
                    # If this reward is always negative due to market conditions, the learner might avoid trading.
                    
                    # Let's use a very basic reward for now to just get trades, then refine.
                    # If `returns.iloc[i + 5]` is positive, reward for buy/long, negative for sell/short
                    # If `returns.iloc[i + 5]` is negative, reward for sell/short, negative for buy/long
                    # This requires knowing the action taken on state `s` when calculating the reward for `s, r`.
                    
                    # The QLearner.query(s,r) is meant to receive the reward *after* taking the previous action and arriving at state `s`.
                    # So, the reward 'r' here should be for the action taken at state `states[i-1]`.
                    # This means your reward `returns.iloc[i + 5]` needs to be `returns.iloc[i + 4]` for the previous day's action.
                    # Or, more cleanly, define the reward for *this* state based on an implied optimal action.
                    
                    # Let's simplify and make the reward directly correlated to profitable movement
                    # for the 'optimal' trade at that point based on future prices.
                    # This is slightly cheating as the learner doesn't know the future, but it simplifies learning
                    # to debug the "no trades" issue.

                    # Temporary simplified reward calculation (for debugging learning):
                    current_price_t = prices[symbol].loc[df_indicators.index[i]]
                    # Ensure the future price exists
                    if i + 5 < len(prices.loc[df_indicators.index]):
                        future_price_t5 = prices[symbol].loc[df_indicators.index[i + 5]]
                        price_change_5d = (future_price_t5 - current_price_t) / current_price_t

                        # Reward for going long (action 2): positive if price goes up
                        # Reward for going short (action 0): positive if price goes down
                        # Reward for holding (action 1): 0
                        # These are ideal rewards. The learner's actions will then be evaluated against these.

                        # A better approach (more standard reinforcement learning):
                        # Reward should be the profit/loss from the action taken *at the previous step*.
                        # For now, let's keep your original structure but note that the reward itself
                        # might need substantial refinement.
                        r = returns.iloc[i] * 1000 if i < len(returns) else 0 # Tie reward to the current state's return (if looking ahead from THIS day)
                                                                            # Or, if `returns` is already aligned, just use `returns.iloc[i+5]` as you had it.
                                                                            # Let's use returns.iloc[i+5] as you had it.
                        # Original: r = returns.iloc[i + 5] * 1000
                        # Let's ensure this is not trying to access out of bounds for `i+5`
                        if (i + 5) < len(returns):
                            r = returns.iloc[i + 5] * 1000
                        else:
                            r = 0 # No future return to observe

                        action = self.learner.query(s, r) # This will take action, and learn from prev state + reward

                    else: # No future data for this state
                        r = 0
                        action = self.learner.query(s, r) # Still query to get an action, but with no reward.
                    
                    # Store (state, action, next_state, reward) for Dyna-Q if applicable
                    # if self.learner.dyna > 0 and i < len(states) - 1:
                    #     experience_tuples.append((states[i], action, states[i+1], r))
        
        # If dyna is enabled, run planning steps after epochs
        # if self.learner.dyna > 0:
        #     # You would need to store unique (s,a) pairs and their (s',r) outcomes
        #     # and then randomly sample from them for Dyna updates.
        #     # This requires a model of the environment.
        #     # For now, let's assume your QLearner handles Dyna internally if `dyna` is set.
        pass # end of add_evidence

    # Origin 
    # def testPolicy(self, symbol="IBM", sd=dt.datetime(2009,1,1), ed=dt.datetime(2010,1,1), sv=100000):
    #     dates = pd.date_range(sd, ed)
    #     prices_all = ut.get_data([symbol], dates)
    #     prices = prices_all[[symbol]].ffill().bfill()

    #     # Compute indicators
    #     bbp = indicators.bollinger_bands_percentage(prices)
    #     macd = indicators.macd_histogram(prices)
    #     rsi = indicators.rsi(prices)

    #     # Normalize
    #     bbp = (bbp - bbp.mean()) / bbp.std()
    #     macd = (macd - macd.mean()) / macd.std()
    #     rsi = (rsi - rsi.mean()) / rsi.std()

    #     df_indicators = pd.concat([bbp, macd, rsi], axis=1).dropna()
    #     df_indicators.columns = ['BBP', 'MACD', 'RSI']

    #     trades = pd.DataFrame(0, index=prices.index, columns=[symbol])
    #     holdings = 0

    #     states = self._compute_states(df_indicators)

    #     for i, date in enumerate(df_indicators.index):
    #         s = states[i]
    #         action = self.learner.querysetstate(s)

    #         if action == 0:  # sell
    #             if holdings == 1000:
    #                 trades.at[date, symbol] = -2000
    #                 holdings = -1000
    #             elif holdings == 0:
    #                 trades.at[date, symbol] = -1000
    #                 holdings = -1000
    #         elif action == 2:  # buy
    #             if holdings == -1000:
    #                 trades.at[date, symbol] = 2000
    #                 holdings = 1000
    #             elif holdings == 0:
    #                 trades.at[date, symbol] = 1000
    #                 holdings = 1000
    #         # action == 1 → hold

    #     return trades

    # def _compute_states(self, df):
    #     # Discretize each indicator into 10 bins and create composite state
    #     bins = 10
    #     bbp_bins = pd.qcut(df['BBP'], bins, labels=False, duplicates='drop')
    #     macd_bins = pd.qcut(df['MACD'], bins, labels=False, duplicates='drop')
    #     rsi_bins = pd.qcut(df['RSI'], bins, labels=False, duplicates='drop')

    #     states = (bbp_bins * 100) + (macd_bins * 10) + rsi_bins
    #     return states.fillna(0).astype(int).values



    # Fix
    def testPolicy(self, symbol="IBM", sd=dt.datetime(2009,1,1), ed=dt.datetime(2010,1,1), sv=100000):
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)
        prices = prices_all[[symbol]].ffill().bfill()

        # --- CRITICAL CHANGE 3: Compute the SAME indicators as ManualStrategy ---
        sma_ratio = indicators.price_sma_ratio(prices, window=self.sma_window)
        bbp = indicators.bollinger_bands_percentage(prices, window=self.bb_window)
        macd_hist = indicators.macd_histogram(prices, fast_period=self.macd_fast,
                                             slow_period=self.macd_slow,
                                             signal_period=self.macd_signal)

        # Normalize the indicators (consistent with add_evidence)
        sma_ratio_norm = (sma_ratio - sma_ratio.mean()) / sma_ratio.std()
        bbp_norm = (bbp - bbp.mean()) / bbp.std()
        macd_hist_norm = (macd_hist - macd_hist.mean()) / macd_hist.std()

        # Fill NaNs created by std=0 or initial rolling window for robustness
        sma_ratio_norm = sma_ratio_norm.replace([np.inf, -np.inf], np.nan).fillna(0)
        bbp_norm = bbp_norm.replace([np.inf, -np.inf], np.nan).fillna(0)
        macd_hist_norm = macd_hist_norm.replace([np.inf, -np.inf], np.nan).fillna(0)

        # --- CRITICAL CHANGE 4: Concatenate and name columns consistently ---
        df_indicators = pd.concat([sma_ratio_norm, bbp_norm, macd_hist_norm], axis=1).dropna()
        df_indicators.columns = ['SMA_Ratio', 'BBP', 'MACD_Hist']

        trades = pd.DataFrame(0, index=prices.index, columns=[symbol])
        holdings = 0

        states = self._compute_states(df_indicators)

        for i, date in enumerate(df_indicators.index):
            s = states[i]
            # In testPolicy, we should NOT pass a reward.
            # Your QLearner.querysetstate(s) should simply return the best action for state `s`
            # based on the learned Q-table, WITHOUT updating the table.
            # If QLearner.query(s) does this, use that. If querysetstate does it, it's fine.
            action = self.learner.querysetstate(s) # Assuming this does not learn/update

            # Translate action to trades
            # 0: SELL, 1: HOLD, 2: BUY
            
            # Potential actions (relative to current holdings):
            # If current_holdings is 1000 (long):
            #   - action=0 (SELL/Short): shares = -2000 (go to -1000)
            #   - action=1 (HOLD): shares = 0 (stay at 1000)
            #   - action=2 (BUY/Long): shares = 0 (stay at 1000, can't buy more)
            # If current_holdings is 0 (out):
            #   - action=0 (SELL/Short): shares = -1000 (go to -1000)
            #   - action=1 (HOLD): shares = 0 (stay at 0)
            #   - action=2 (BUY/Long): shares = 1000 (go to 1000)
            # If current_holdings is -1000 (short):
            #   - action=0 (SELL/Short): shares = 0 (stay at -1000, can't sell more)
            #   - action=1 (HOLD): shares = 0 (stay at -1000)
            #   - action=2 (BUY/Long): shares = 2000 (go to 1000)

            # Let's make this explicit based on desired final position
            # Target position map: action 0 -> -1000, action 1 -> 0, action 2 -> 1000
            target_holdings = 0
            if action == 0: # Target short
                target_holdings = -1000
            elif action == 2: # Target long
                target_holdings = 1000
            # else: target_holdings = 0 (for action 1)

            shares_to_trade = target_holdings - holdings

            if shares_to_trade != 0:
                trades.at[date, symbol] = shares_to_trade
                holdings += shares_to_trade # Update holdings for next iteration


        return trades

    def _compute_states(self, df):
        # Discretize each indicator into 10 bins and create composite state
        bins = 10

        # --- CRITICAL CHANGE 5: Update binning to use the new indicator columns ---
        # Handle duplicates='drop' for qcut, but also consider cases where all values might be the same
        # If all values are the same, qcut with duplicates='drop' can lead to fewer bins or errors.
        # A simple check for unique values can help.
        
        # Ensure 'dropna()' is handled effectively before passing to qcut
        # The df already has dropna() applied, so this should be fine.
        
        # Safely apply qcut: if unique values < bins, reduce bins for that indicator
        def safe_qcut(series, num_bins):
            unique_vals = series.nunique()
            actual_bins = min(num_bins, unique_vals) if unique_vals > 0 else 1 # Ensure at least 1 bin if data exists
            if actual_bins <= 1: # If only one unique value, or no data, map all to 0
                return pd.Series(0, index=series.index)
            try:
                # Use retbins=True to inspect bin edges for debugging if needed
                return pd.qcut(series, actual_bins, labels=False, duplicates='drop')
            except ValueError as e:
                print(f"Warning: qcut failed for series with {unique_vals} unique values. Error: {e}")
                return pd.Series(0, index=series.index) # Fallback to all zeros
        
        sma_bins = safe_qcut(df['SMA_Ratio'], bins)
        bbp_bins = safe_qcut(df['BBP'], bins)
        macd_bins = safe_qcut(df['MACD_Hist'], bins)

        # --- CRITICAL CHANGE 6: Update state calculation based on new columns ---
        # The factors (100, 10, 1) should ensure unique states up to 1000
        states = (sma_bins * (bins * bins)) + (bbp_bins * bins) + macd_bins # (e.g., bin * bin for first, bin for second, 1 for third)
        
        # Ensure state values are integers and handle any remaining NaNs (though dropna() on df_indicators should prevent this)
        return states.fillna(0).astype(int).values