import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

from marketsimcode import compute_portvals
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
from util import get_data


def normalize(df):
    return df / df.iloc[0]


def experiment2():
    symbol = "JPM"
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    sv = 100000

    # Initialize learners
    manual = ManualStrategy()
    learner = StrategyLearner(verbose=False, impact=0.0)

    # Get in-sample prices
    prices = get_data([symbol], pd.date_range(start_date, end_date))
    prices = prices[[symbol]]

    # Manual Strategy
    manual_trades = manual.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=sv)
    manual_portvals = compute_portvals(manual_trades, start_val=sv)
    manual_portvals = normalize(manual_portvals)

    # Strategy Learner
    learner.addEvidence(symbol=symbol, sd=start_date, ed=end_date, sv=sv)
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
    plt.savefig("experiment2.png")
    plt.show()

    # Performance statistics
    print("Manual Strategy:")
    print(f"Cumulative Return: {manual_portvals.iloc[-1] / manual_portvals.iloc[0] - 1:.4f}")
    print(f"Standard Deviation: {manual_portvals.std()[0]:.4f}")
    print(f"Mean Daily Return: {manual_portvals.pct_change().mean()[0]:.4f}")

    print("\nStrategy Learner:")
    print(f"Cumulative Return: {learner_portvals.iloc[-1] / learner_portvals.iloc[0] - 1:.4f}")
    print(f"Standard Deviation: {learner_portvals.std()[0]:.4f}")
    print(f"Mean Daily Return: {learner_portvals.pct_change().mean()[0]:.4f}")


if __name__ == "__main__":
    experiment2()
