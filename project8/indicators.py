# import numpy as np
# import pandas as pd
# from util import get_data, symbol_to_path
# import datetime as dt
# import matplotlib.pyplot as plt
# start_date = dt.datetime(2008, 1, 1)
# end_date = dt.datetime(2009, 12, 31)
# stock = ['JPM']
# df = get_data(stock,pd.date_range(start_date, end_date), addSPY=False)
# df = df.dropna()
# #print (df)

# def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
#     import matplotlib.pyplot as plt

#     """Plot stock prices with a custom title and meaningful axis labels."""
#     """DO NOT use in code submitted for grading"""
#     ax = df.plot(title=title, fontsize=12)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)



# def sma(df, days, plot=False, combine=False):
#     '''

#     :param df: stocks' sma to be calculated
#     :param days: windows
#     :param plot: whether to plot and save data
#     :param combine: whether to combine orignial data to sma data
#     :return: sma dataframe
#     '''
#     smadf = df.rolling(days).mean()
#     #smadf = smadf/(df.iloc[0])
#     if combine == True:
#         df = pd.concat([df, smadf], axis = 1)

#         #print(thisstocks)
#         df.columns = ['JPM', 'SMA']

#     if plot == True:
#         plot_data(df, title="Simple Moving Average of JPM (Window={})".format(days), xlabel="Date", ylabel="Average Prices")
#         plt.savefig(".//images//Figure1.png")
#         plt.show()
#         plt.close()
#     #print(df)
#     return (smadf)

# def bollingerband(df, days, plot=False, combine=False):
#     stddf = df.rolling(days).std()
#     sma = df.rolling(days).mean()
#     upperband = sma + 2 * stddf
#     lowerband = sma - 2 * stddf
#     bb = pd.concat([lowerband, upperband], axis=1)
#     bb.columns = ['lowerband', 'upperband']
#     printed = bb

#     bbp = (df - sma)/(2 * stddf)

#     if combine == True:
#         printed = pd.concat([df, bb], axis=1)
#         thisstocks = stock
#         thisstocks.append('lowerband')
#         thisstocks.append('upperband')
#         #print(thisstocks)
#         printed.columns = thisstocks

#     if plot == True:
#         plot_data(printed, title="Bollinger Band of {} (Window={})".format(stock[0], days), xlabel="Date",
#                   ylabel="bollingerband prices")
#         plt.savefig(".//images//Figure2.png")
#         plt.show()
#         plt.close()

#     return(bbp)

# def get_momentum(prices, days, plot=False):

#     momentum = prices/prices.shift(days-1) - 1

#     if plot == True:
#         plot_data(momentum, title="Momentum of {} (Window={})".format(stock[0], days), xlabel="Date",
#                   ylabel="momentum ratio")
#         plt.savefig(".//images//Figure3-momentum.png")
#         plt.show()
#         plt.close()
#     return momentum
# def EMA(prices, days):
#     emadf = prices * 2

#     for i in range(prices.shape[0]-days + 1):
#         tempema = prices.iloc[i]
#         for j in range(days):
#             tempema =prices.iloc[i+j] * (2.0/(j + 2)) + tempema * (1-2.0/(j+2))
#         emadf.iloc[i + days - 1] = tempema
#     emadf.iloc[0: (days - 1)] = np.nan

#     return emadf

# def PPI(prices, plot=False):
#     PPI = (EMA(prices, 12) - EMA(prices, 26))/EMA(prices, 26)
#     PPI[0:26] = np.nan
#     if plot == True:
#         plot_data(PPI, title="Percentage Price Oscillator of JPM (12 days/26 days)", xlabel="Date", ylabel="Percentage Price Oscillator")
#         plt.savefig(".//images//Figure4-PPI.png")
#         plt.show()
#         plt.close()
#     return PPI

# def GoldenCross(prices, day1, day2, plot=False):
#     smaday1 = sma(prices, day1)
#     smaday2 = sma(prices, day2)
#     gc = pd.concat([smaday1, smaday2], axis=1)

#     gc.columns = ['{}-day SMA'.format(day1), '{}-day SMA'.format(day2)]
#     #gc['difference'] = gc['{}-day SMA'.format(day1)] - gc['{}-day SMA'.format(day2)]

#     if plot == True:
#         plot_data(gc, title="Golden Cross of JPM ({} days/{} days)".format(day1, day2), xlabel="Date", ylabel="SMA difference between two windows")
#         plt.savefig(".//images//Figure5-GC.png")
#         plt.show()
#         plt.close()
#     return gc


# def author():
#     return 'twu411'


import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from util import get_data

def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "awang758"

def get_rolling_mean(values, window):

    return values.rolling(window=window).mean()

def get_rolling_std(values, window):

    return values.rolling(window=window).std()

def bollinger_bands_percentage(prices, window=20):
    if isinstance(prices, pd.DataFrame):
        price_series = prices.iloc[:, 0]
    else:
        price_series = prices

    sma = get_rolling_mean(price_series, window)
    rstd = get_rolling_std(price_series, window)

    upper_band = sma + (2 * rstd)
    lower_band = sma - (2 * rstd)

    bbp = (price_series - lower_band) / (upper_band - lower_band)
    return bbp.rename('Bollinger %B')


def cci(prices, window=20):
    if isinstance(prices, pd.DataFrame):
        try:
            high = get_data([prices.columns[0]], prices.index, addSPY=False, colname='High').iloc[:, 0]
            low = get_data([prices.columns[0]], prices.index, addSPY=False, colname='Low').iloc[:, 0]
            close = prices.iloc[:, 0] # Assuming this is 'Adj Close'
        except Exception:
            print(f"Warning: High/Low data not found for CCI. Using Adj Close for High/Low values. CCI calculation may be less accurate.")
            high = prices.iloc[:, 0]
            low = prices.iloc[:, 0]
            close = prices.iloc[:, 0]
            
    else:
        print(f"Warning: Only Adj Close price provided for CCI. Using Adj Close for High/Low values. CCI calculation may be less accurate.")
        high = prices
        low = prices
        close = prices

    typical_price = (high + low + close) / 3
    sma_tp = get_rolling_mean(typical_price, window)
    
    mean_deviation = get_rolling_mean(abs(typical_price - sma_tp), window)
    
    cci_values = (typical_price - sma_tp) / (0.015 * mean_deviation)
    cci_values = cci_values.replace([np.inf, -np.inf], np.nan)

    return cci_values.rename('CCI')
def macd_histogram(prices, fast_period=12, slow_period=26, signal_period=9):

    if isinstance(prices, pd.DataFrame):
        price_series = prices.iloc[:, 0]
    else:
        price_series = prices

    ema_fast = price_series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = price_series.ewm(span=slow_period, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    macd_hist = macd_line - signal_line
    return macd_hist.rename('MACD Histogram')

def rsi(prices, window=14):
    if isinstance(prices, pd.DataFrame):
        price_series = prices.iloc[:, 0]
    else:
        price_series = prices

    deltas = price_series.diff()
    gains = deltas.clip(lower=0)
    losses = -deltas.clip(upper=0)
    avg_gain = gains.ewm(span=window, adjust=False).mean()
    avg_loss = losses.ewm(span=window, adjust=False).mean()

    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    rsi_values = 100 - (100 / (1 + rs))

    rsi_values = pd.Series(rsi_values, index=price_series.index).replace([np.inf, -np.inf], np.nan)

    return rsi_values.rename('RSI')

def price_sma_ratio(prices, window=20):
    if isinstance(prices, pd.DataFrame):
        price_series = prices.iloc[:, 0]
    else:
        price_series = prices

    sma = get_rolling_mean(price_series, window)
    
    ratio = price_series / sma
    return ratio.rename('Price/SMA Ratio')