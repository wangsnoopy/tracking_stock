import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

from marketsimcode import compute_portvals
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
from util import get_data


def normalize(df):
    return df / df.iloc[0]


def run():
    symbol = "JPM"
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    sv = 100000

    # Initialize learners
    manual = ManualStrategy()
    learner = StrategyLearner(verbose=False, impact=0.0)

    # Get in-sample prices (optional here, but you keep for reference)
    prices = get_data([symbol], pd.date_range(start_date, end_date))
    prices = prices[[symbol]]

    # Manual Strategy
    manual_trades = manual.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=sv)
    manual_portvals = compute_portvals(manual_trades, start_val=sv)
    manual_portvals = normalize(manual_portvals)

    # Strategy Learner
    learner.add_evidence(symbol=symbol, sd=start_date, ed=end_date, sv=sv)
    learner_trades = learner.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=sv)
    learner_portvals = compute_portvals(learner_trades, start_val=sv)
    learner_portvals = normalize(learner_portvals)

    # Plot results
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

    # Performance statistics
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
