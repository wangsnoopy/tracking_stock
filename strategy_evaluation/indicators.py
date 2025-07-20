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