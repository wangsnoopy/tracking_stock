# import pandas as pd
# import numpy as np
# import datetime as dt
# from util import get_data
# import indicators

# class ManualStrategy:
#     def __init__(self, verbose=False):
#         self.verbose = verbose

#     def author(self):
#         return 'awang758'

#     def testPolicy(self, symbol="JPM", sd=dt.datetime(2008, 1, 1),
#                    ed=dt.datetime(2009, 12, 31), sv=100000):
#         # Get price data
#         dates = pd.date_range(sd, ed)
#         prices_all = get_data([symbol], dates)
#         prices = prices_all[[symbol]].ffill().bfill()

#         # Compute indicators (all return Series)
#         sma_ratio = indicators.price_sma_ratio(prices, window=20)
#         bbp = indicators.bollinger_bands_percentage(prices, window=20)
#         macd_hist = indicators.macd_histogram(prices)

#         # Initialize trades DataFrame
#         trades = pd.DataFrame(data=0, index=prices.index, columns=[symbol])
#         position = 0  # 0 = no position, 1000 = long, -1000 = short

#         for i in range(1, len(prices)):
#             date = prices.index[i]

#             sma_val = sma_ratio.loc[date]
#             bbp_val = bbp.loc[date]
#             macd_val = macd_hist.loc[date]

#             long_signal = (sma_val < 0.99) and (bbp_val < 0.2) and (macd_val > 0.5)
#             short_signal = (sma_val > 1.01) and (bbp_val > 0.8) and (macd_val < -0.5)

#             if position == 0:
#                 if long_signal:
#                     trades.loc[date] = 1000
#                     position = 1000
#                 elif short_signal:
#                     trades.loc[date] = -1000
#                     position = -1000
#             elif position == 1000:
#                 if short_signal:
#                     trades.loc[date] = -2000  # Sell 1000 + short 1000
#                     position = -1000
#                 elif not long_signal:
#                     trades.loc[date] = -1000  # Close long
#                     position = 0
#             elif position == -1000:
#                 if long_signal:
#                     trades.loc[date] = 2000  # Cover 1000 + long 1000
#                     position = 1000
#                 elif not short_signal:
#                     trades.loc[date] = 1000  # Close short
#                     position = 0

#         return trades



import pandas as pd
import numpy as np
import datetime as dt
import indicators
import marketsimcode as msc
import util as ut


class ManualStrategy:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.symbol = None
        self.sd = None
        self.ed = None

    def author(self):
        return 'awang412'  # Replace with your Georgia Tech username

    def testPolicy(self, symbol='JPM',
                   sd=dt.datetime(2008, 1, 1),
                   ed=dt.datetime(2009, 12, 31),
                   sv=100000):

        self.symbol = symbol
        self.sd = sd
        self.ed = ed

        # Get price data
        prices = ut.get_data([symbol], pd.date_range(sd, ed))
        prices = prices[[symbol]]
        prices.ffill(inplace=True)
        prices.bfill(inplace=True)

        # Get indicators
        # sma_ratio = indicators.get_sma_ratio(prices, window=20)
        # bbp = indicators.get_bollinger_band_percent(prices, window=20)
        # macd_hist = indicators.get_momentum(prices, window=10)
        sma_ratio = indicators.price_sma_ratio(prices, window=20)
        bbp = indicators.bollinger_bands_percentage(prices, window=20)
        macd_hist = indicators.macd_histogram(prices)

        # Create a DataFrame for trades
        trades = pd.DataFrame(index=prices.index, data=0, columns=[symbol])

        position = 0  # +1000 for long, -1000 for short, 0 for neutral

        for i in range(1, prices.shape[0]):
            date = prices.index[i]
            signal = 0

            # BUY SIGNAL
            if bbp.iloc[i][symbol] < 0.2 and macd_hist.iloc[i][symbol] > 0 and sma_ratio.iloc[i][symbol] < 0.95:
                if position <= 0:
                    signal = 1000 - position
                    position = 1000

            # SELL SIGNAL
            elif bbp.iloc[i][symbol] > 0.8 and macd_hist.iloc[i][symbol] < 0 and sma_ratio.iloc[i][symbol] > 1.05:
                if position >= 0:
                    signal = -1000 - position
                    position = -1000

            # EXIT SIGNAL (mean reversion / closing position)
            elif abs(macd_hist.iloc[i][symbol]) < 0.01 and abs(sma_ratio.iloc[i][symbol] - 1.0) < 0.01:
                if position != 0:
                    signal = -position
                    position = 0

            trades.loc[date, symbol] = signal

        return trades


