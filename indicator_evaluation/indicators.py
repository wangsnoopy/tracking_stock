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
    return "awang758" # Replace with your actual GT username

# Helper functions (from previous steps, included for completeness)
def get_rolling_mean(values, window):
    """Return rolling mean of given values, using specified window size."""
    return values.rolling(window=window).mean()

def get_rolling_std(values, window):
    """Return rolling standard deviation of given values, using specified window size."""
    return values.rolling(window=window).std()

# --- Indicator 1: Bollinger Bands Percentage (%B) ---
def bollinger_bands_percentage(prices, window=20):
    """
    Calculates the Bollinger Bands Percentage (%B).
    Returns a Series of Bollinger Bands Percentage values.
    """
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

# --- Indicator 2: Commodity Channel Index (CCI) ---
def cci(prices, window=20):
    """
    Calculates the Commodity Channel Index (CCI).
    CCI = (Typical Price - SMA of Typical Price) / (0.015 * Mean Deviation)

    Typical Price (TP) = (High + Low + Close) / 3
    Mean Deviation = SMA of absolute difference between TP and its SMA

    Returns a Series of CCI values.
    """
    if isinstance(prices, pd.DataFrame):
        # We need High, Low, and Close for CCI. Assuming prices DataFrame contains these.
        # If get_data only returns Adj Close, you'll need to adapt how you fetch data
        # or simplify this indicator for Adj Close only (less accurate CCI).
        # For this example, let's assume 'High', 'Low', 'Adj Close' are available.
        # If not, you'll need to fetch them using get_data with appropriate columns.
        try:
            high = get_data([prices.columns[0]], prices.index, addSPY=False, colname='High').iloc[:, 0]
            low = get_data([prices.columns[0]], prices.index, addSPY=False, colname='Low').iloc[:, 0]
            close = prices.iloc[:, 0] # Assuming this is 'Adj Close'
        except Exception:
            # Fallback if High/Low not directly available, use Adj Close for simplicity
            print(f"Warning: High/Low data not found for CCI. Using Adj Close for High/Low values. CCI calculation may be less accurate.")
            high = prices.iloc[:, 0]
            low = prices.iloc[:, 0]
            close = prices.iloc[:, 0]
            
    else: # If prices is a Series (only Adj Close)
        print(f"Warning: Only Adj Close price provided for CCI. Using Adj Close for High/Low values. CCI calculation may be less accurate.")
        high = prices
        low = prices
        close = prices

    typical_price = (high + low + close) / 3
    sma_tp = get_rolling_mean(typical_price, window)
    
    mean_deviation = get_rolling_mean(abs(typical_price - sma_tp), window)
    
    # Handle division by zero for mean_deviation
    cci_values = (typical_price - sma_tp) / (0.015 * mean_deviation)
    cci_values = cci_values.replace([np.inf, -np.inf], np.nan) # Replace inf with NaN

    return cci_values.rename('CCI')

# --- Indicator 3: MACD (Refactored to return single vector - Histogram) ---
def macd_histogram(prices, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculates the MACD Histogram.
    MACD Line = 12-period EMA - 26-period EMA
    Signal Line = 9-period EMA of MACD Line
    MACD Histogram = MACD Line - Signal Line

    Returns a Series of MACD Histogram values.
    """
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

# --- Indicator 4: Relative Strength Index (RSI) ---
def rsi(prices, window=14):
    """
    Calculates the Relative Strength Index (RSI).
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss

    Returns a Series of RSI values.
    """
    if isinstance(prices, pd.DataFrame):
        price_series = prices.iloc[:, 0]
    else:
        price_series = prices

    deltas = price_series.diff()
    gains = deltas.clip(lower=0)
    losses = -deltas.clip(upper=0) # Make losses positive

    # Use EWMA for average gain/loss for more common RSI calculation
    # Or SMA if strictly adhering to your previous reference (but EWMA is standard for RSI)
    avg_gain = gains.ewm(span=window, adjust=False).mean()
    avg_loss = losses.ewm(span=window, adjust=False).mean()

    # Handle division by zero
    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    rsi_values = 100 - (100 / (1 + rs))
    
    # Replace inf values with NaN for proper plotting/handling
    rsi_values = pd.Series(rsi_values, index=price_series.index).replace([np.inf, -np.inf], np.nan)

    return rsi_values.rename('RSI')

# --- Indicator 5: Price/SMA Ratio (Custom Indicator) ---
def price_sma_ratio(prices, window=20):
    """
    Calculates the ratio of the current price to its Simple Moving Average (SMA).
    This can be interpreted as a momentum-like indicator.
    Returns a Series of Price/SMA Ratio values.
    """
    if isinstance(prices, pd.DataFrame):
        price_series = prices.iloc[:, 0]
    else:
        price_series = prices

    sma = get_rolling_mean(price_series, window)
    
    ratio = price_series / sma
    return ratio.rename('Price/SMA Ratio')