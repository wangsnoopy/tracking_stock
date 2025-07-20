import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

from marketsimcode import compute_portvals
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
from util import get_data


def trades_to_orders(trades_df):
    """
    Convert trades DataFrame (daily holdings) to orders DataFrame.
    """
    orders = []

    for symbol in trades_df.columns:
        prev_shares = 0
        for date, shares in trades_df[symbol].iteritems():
            delta = shares - prev_shares
            if delta > 0:
                orders.append({'Date': date, 'Symbol': symbol, 'Order': 'BUY', 'Shares': delta})
            elif delta < 0:
                orders.append({'Date': date, 'Symbol': symbol, 'Order': 'SELL', 'Shares': -delta})
            prev_shares = shares

    orders_df = pd.DataFrame(orders)
    if not orders_df.empty:
        orders_df.set_index('Date', inplace=True)
        orders_df.index = pd.to_datetime(orders_df.index)
        orders_df.sort_index(inplace=True)
    else:
        orders_df = pd.DataFrame(columns=['Symbol', 'Order', 'Shares'])

    return orders_df


def normalize(df):
    return df / df.iloc[0]


def run():
    symbol = "JPM"
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    sv = 100000

    manual = ManualStrategy()
    learner = StrategyLearner(verbose=False, impact=0.0)

    manual_trades = manual.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=sv)
    learner.add_evidence(symbol=symbol, sd=start_date, ed=end_date, sv=sv)
    learner_trades = learner.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=sv)

    orders_manual = trades_to_orders(manual_trades)
    orders_learner = trades_to_orders(learner_trades)

    manual_portvals = compute_portvals(orders_manual, start_val=sv)
    learner_portvals = compute_portvals(orders_learner, start_val=sv)

    manual_portvals = normalize(manual_portvals)
    learner_portvals = normalize(learner_portvals)

    plt.figure(figsize=(10, 6))
    plt.plot(manual_portvals, label="Manual Strategy", color='blue')
    plt.plot(learner_portvals, label="Strategy Learner", color='orange')
    plt.title("Experiment 2: Manual Strategy vs. Strategy Learner")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("images/experiment2.png")
    plt.show()

    def print_stats(portvals, label):
        daily_returns = portvals.pct_change().dropna()
        cum_return = portvals.iloc[-1] / portvals.iloc[0] - 1
        std_daily_ret = daily_returns.std()
        mean_daily_ret = daily_returns.mean()

        print(f"{label}:")
        print(f"  Cumulative Return: {cum_return:.4f}")
        print(f"  Std of Daily Return: {std_daily_ret:.4f}")
        print(f"  Mean Daily Return: {mean_daily_ret:.4f}")
        print()

    print_stats(manual_portvals, "Manual Strategy")
    print_stats(learner_portvals, "Strategy Learner")


if __name__ == "__main__":
    run()

