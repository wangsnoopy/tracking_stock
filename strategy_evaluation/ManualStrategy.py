import pandas as pd
import numpy as np
import datetime as dt
from util import get_data
import indicators

class ManualStrategy:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def author(self):
        return 'awang758'

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008, 1, 1),
                   ed=dt.datetime(2009, 12, 31), sv=100000):
        # Get price data
        dates = pd.date_range(sd, ed)
        prices_all = get_data([symbol], dates)
        prices = prices_all[[symbol]].ffill().bfill()

        # Compute indicators
        sma_ratio = indicators.compute_price_sma_ratio(prices, window=20)
        bbp = indicators.compute_bbp(prices, window=20)
        macd_line, signal_line = indicators.compute_macd(prices)

        # Initialize trades DataFrame
        trades = pd.DataFrame(data=0, index=prices.index, columns=[symbol])
        position = 0  # Current position: 0, 1000, -1000

        for i in range(1, len(prices)):
            date = prices.index[i]

            # Get indicator values for the day
            sma_val = sma_ratio.loc[date, symbol]
            bbp_val = bbp.loc[date, symbol]
            macd_val = macd_line.loc[date, symbol]
            signal_val = signal_line.loc[date, symbol]

            # Define entry and exit signals
            long_signal = (sma_val < 0.95) and (bbp_val < 0.0) and (macd_val > signal_val)
            short_signal = (sma_val > 1.05) and (bbp_val > 1.0) and (macd_val < signal_val)

            if position == 0:
                if long_signal:
                    trades.loc[date] = 1000
                    position = 1000
                elif short_signal:
                    trades.loc[date] = -1000
                    position = -1000
            elif position == 1000:
                if short_signal:
                    trades.loc[date] = -2000  # Close long + open short
                    position = -1000
                elif not long_signal:
                    trades.loc[date] = -1000  # Close long
                    position = 0
            elif position == -1000:
                if long_signal:
                    trades.loc[date] = 2000  # Close short + open long
                    position = 1000
                elif not short_signal:
                    trades.loc[date] = 1000  # Close short
                    position = 0

        return trades
