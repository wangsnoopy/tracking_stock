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

# Helper function for plotting, similar to what you provided in reference
def plot_indicator(df_plot, title, ylabel, xlabel="Date", save_path="./images/"):
    plt.figure(figsize=(12, 7))
    ax = df_plot.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    # plt.savefig(f"{save_path}{title.replace(' ', '_').replace('/', '')}.png")
    plt.show()
    plt.close()

def get_rolling_mean(values, window):
    """Return rolling mean of given values, using specified window size."""
    return values.rolling(window=window).mean()

def get_rolling_std(values, window):
    """Return rolling standard deviation of given values, using specified window size."""
    return values.rolling(window=window).std()

def bollinger_bands_percentage(prices, window=20):
    """
    Calculates the Bollinger Bands Percentage (%B).

    %B = (Current Price - Lower Bollinger Band) / (Upper Bollinger Band - Lower Bollinger Band)

    A value of %B > 1 indicates the price is above the upper band (overbought).
    A value of %B < 0 indicates the price is below the lower band (oversold).
    A value of %B = 0.5 indicates the price is at the SMA.

    Parameters:
        prices (pd.Series or pd.DataFrame): Stock prices.
        window (int): The look-back window for SMA and Standard Deviation.

    Returns:
        pd.Series: A Series of Bollinger Bands Percentage values.
    """
    if isinstance(prices, pd.DataFrame):
        # Assuming prices is a DataFrame with a single column (e.g., 'JPM')
        price_series = prices.iloc[:, 0]
    else:
        price_series = prices

    sma = get_rolling_mean(price_series, window)
    rstd = get_rolling_std(price_series, window)

    # Upper and Lower Bollinger Bands
    upper_band = sma + (2 * rstd)
    lower_band = sma - (2 * rstd)

    # Calculate %B
    bbp = (price_series - lower_band) / (upper_band - lower_band)
    
    # Return as a Series with proper name
    return bbp.rename('Bollinger %B')

# Example usage and plotting will be in the main execution block or testproject.py
# For now, let's include a placeholder for direct execution for testing
if __name__ == "__main__":
    symbol = "JPM"
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)

    # Get data
    prices = get_data([symbol], pd.date_range(sd, ed), addSPY=False).dropna()
    prices = prices[[symbol]] # Ensure it's just the symbol we care about

    # Calculate Bollinger Bands Percentage
    bbp_values = bollinger_bands_percentage(prices, window=20)

    # Prepare data for plotting
    plot_df = pd.DataFrame(index=prices.index)
    plot_df['JPM'] = prices[symbol]
    plot_df['Bollinger %B'] = bbp_values

    # Add SMA, Upper Band, Lower Band for context in the plot
    sma = get_rolling_mean(prices[symbol], window=20)
    rstd = get_rolling_std(prices[symbol], window=20)
    upper_band = sma + (2 * rstd)
    lower_band = sma - (2 * rstd)

    plot_df['SMA'] = sma
    plot_df['Upper Band'] = upper_band
    plot_df['Lower Band'] = lower_band

    # Normalize price and SMA for better visualization if desired
    # plot_df['JPM (Normalized)'] = plot_df['JPM'] / plot_df['JPM'].iloc[0]
    # plot_df['SMA (Normalized)'] = plot_df['SMA'] / plot_df['JPM'].iloc[0]
    # plot_df['Upper Band (Normalized)'] = plot_df['Upper Band'] / plot_df['JPM'].iloc[0]
    # plot_df['Lower Band (Normalized)'] = plot_df['Lower Band'] / plot_df['JPM'].iloc[0]


    # Plotting
    # We will use two subplots for clarity: Price with Bands, and %B
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))

    # Plot 1: Price with Bollinger Bands
    ax1.plot(plot_df.index, plot_df['JPM'], label='JPM Price', color='blue')
    ax1.plot(plot_df.index, plot_df['SMA'], label='SMA (20)', color='orange', linestyle='--')
    ax1.plot(plot_df.index, plot_df['Upper Band'], label='Upper Band', color='red', linestyle=':')
    ax1.plot(plot_df.index, plot_df['Lower Band'], label='Lower Band', color='green', linestyle=':')
    ax1.set_ylabel("Price")
    ax1.set_title("JPM Price with Bollinger Bands")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Bollinger Bands Percentage (%B)
    ax2.plot(plot_df.index, plot_df['Bollinger %B'], label='Bollinger %B', color='purple')
    ax2.axhline(1.0, color='red', linestyle='--', label='Overbought (1.0)')
    ax2.axhline(0.0, color='green', linestyle='--', label='Oversold (0.0)')
    ax2.set_xlabel("Date")
    ax2.set_ylabel("%B Value")
    ax2.set_title("Bollinger Bands Percentage (%B)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    # plt.savefig("./images/Bollinger_Bands_Percentage.png")
    plt.show()
    plt.close()

    print("Generated Bollinger Bands Percentage plot.")