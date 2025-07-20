import numpy as np
import pandas as pd
from util import get_data, plot_data


def author():
    """Returns the GT username."""
    return "awang758"


def gtid():
    """Returns the GT ID."""
    return 904081341


def study_group():
    return "awang758"


def compute_portvals(
        orders,
        start_val=1000000,
        commission=0.0,
        impact=0.0,
):
    """
    Computes portfolio values given orders as a DataFrame.

    :param orders: DataFrame with columns ['Symbol', 'Order', 'Shares'], indexed by date
    :param start_val: initial cash value
    :param commission: fixed commission per trade
    :param impact: market impact per trade
    :return: portfolio values as a DataFrame with column 'Portfolio Value'
    """

    # Infer date range from orders DataFrame
    start_date = orders.index.min()
    end_date = orders.index.max()

    unique_stocks = list(orders['Symbol'].unique())

    # Get price data for all symbols in date range, drop SPY if present
    prices = get_data(unique_stocks, pd.date_range(start_date, end_date)).drop(columns=["SPY"], errors='ignore')
    prices["Cash"] = 1.0  # Cash column for portfolio cash position

    # Initialize trades DataFrame with zeros
    trades = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    # Populate trades based on orders
    for date, order in orders.iterrows():
        symbol = order['Symbol']
        shares = order['Shares']
        position_factor = 1 if order['Order'].upper() == "BUY" else -1

        # Update the trades for symbol shares
        trades.at[date, symbol] += shares * position_factor

        # Calculate cash impact for the trade including commission and market impact
        price = prices.at[date, symbol]
        cash_impact = (-position_factor) * price * shares
        market_impact_fee = impact * abs(cash_impact)
        trades.at[date, "Cash"] += cash_impact - market_impact_fee - commission

    # Calculate holdings by cumulative sum of trades
    holdings = trades.cumsum()
    holdings.at[holdings.index[0], "Cash"] += start_val  # Add initial cash to first day

    # Calculate total portfolio value (holdings * prices)
    values = holdings * prices
    port_vals = values.sum(axis=1)

    return pd.DataFrame(port_vals, columns=['Portfolio Value'])


def test_code():
    """Helper function to test compute_portvals."""
    orders_file = "./orders/orders-01.csv"
    start_val = 1000000

    # Read orders CSV into DataFrame
    orders_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True)

    # Compute portfolio values
    portvals = compute_portvals(orders_df, start_val=start_val, commission=9.95, impact=0.005)

    # Calculate statistics
    daily_returns = portvals['Portfolio Value'].pct_change().dropna()
    cum_return = (portvals['Portfolio Value'].iloc[-1] / portvals['Portfolio Value'].iloc[0]) - 1
    avg_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std(ddof=1)
    sharpe_ratio = np.sqrt(252) * (avg_daily_return / std_daily_return)

    start_date = portvals.index.min()
    end_date = portvals.index.max()

    dates = pd.date_range(start_date, end_date)
    spy_prices = get_data(['$SPX'], dates)[['$SPX']]
    spy_prices = spy_prices.loc[portvals.index]
    spy_prices = spy_prices.fillna(method='ffill').fillna(method='bfill')

    daily_returns_spy = spy_prices.pct_change().dropna()

    cum_return_spy = (spy_prices.iloc[-1, 0] / spy_prices.iloc[0, 0]) - 1
    avg_daily_return_spy = daily_returns_spy.mean()[0]
    std_daily_return_spy = daily_returns_spy.std()[0]
    sharpe_ratio_spy = (avg_daily_return_spy / std_daily_return_spy) * np.sqrt(252)

    # Print results
    print(f"Date Range: {start_date} to {end_date}")
    print("-" * 30)
    print(f"Sharpe Ratio of Fund: {sharpe_ratio:.4f}")
    print(f"Sharpe Ratio of $SPX: {sharpe_ratio_spy:.4f}")
    print()
    print(f"Cumulative Return of Fund: {cum_return:.4f}")
    print(f"Cumulative Return of $SPX: {cum_return_spy:.4f}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_return:.4f}")
    print(f"Standard Deviation of $SPX: {std_daily_return_spy:.4f}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_return:.4f}")
    print(f"Average Daily Return of $SPX: {avg_daily_return_spy:.4f}")
    print()
    print(f"Final Portfolio Value: {portvals['Portfolio Value'].iloc[-1]:.2f}")


if __name__ == "__main__":
    test_code()
