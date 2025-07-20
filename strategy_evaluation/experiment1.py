import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
import marketsimcode as ms
import util as ut


def run_experiment(symbol='JPM', start_in=dt.datetime(2008, 1, 1), end_in=dt.datetime(2009, 12, 31),
                   start_out=dt.datetime(2010, 1, 1), end_out=dt.datetime(2011, 12, 31),
                   start_val=100000):

    # Manual Strategy
    ms_strategy = ManualStrategy()
    ms_strategy.add_evidence(symbol=symbol, sd=start_in, ed=end_in)
    trades_manual_in = ms_strategy.testPolicy(symbol=symbol, sd=start_in, ed=end_in)
    trades_manual_out = ms_strategy.testPolicy(symbol=symbol, sd=start_out, ed=end_out)

    portvals_manual_in = ms.compute_portvals(trades_manual_in, start_val=start_val)
    portvals_manual_out = ms.compute_portvals(trades_manual_out, start_val=start_val)

    # Strategy Learner
    sl = StrategyLearner(verbose=False, impact=0.0)
    sl.add_evidence(symbol=symbol, sd=start_in, ed=end_in)
    trades_learner_in = sl.testPolicy(symbol=symbol, sd=start_in, ed=end_in)
    trades_learner_out = sl.testPolicy(symbol=symbol, sd=start_out, ed=end_out)

    portvals_learner_in = ms.compute_portvals(trades_learner_in, start_val=start_val)
    portvals_learner_out = ms.compute_portvals(trades_learner_out, start_val=start_val)

    # Normalize
    portvals_manual_in /= portvals_manual_in.iloc[0]
    portvals_learner_in /= portvals_learner_in.iloc[0]
    portvals_manual_out /= portvals_manual_out.iloc[0]
    portvals_learner_out /= portvals_learner_out.iloc[0]

    # Plot In-sample
    plt.figure(figsize=(12, 6))
    plt.plot(portvals_manual_in, label='Manual Strategy')
    plt.plot(portvals_learner_in, label='Strategy Learner')
    plt.title(f"In-Sample Performance: {symbol} (2008–2009)")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend()
    plt.grid()
    plt.savefig("experiment1_in_sample.png")
    plt.close()

    # Plot Out-of-sample
    plt.figure(figsize=(12, 6))
    plt.plot(portvals_manual_out, label='Manual Strategy')
    plt.plot(portvals_learner_out, label='Strategy Learner')
    plt.title(f"Out-of-Sample Performance: {symbol} (2010–2011)")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend()
    plt.grid()
    plt.savefig("experiment1_out_sample.png")
    plt.close()

    # Print performance stats
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

    print_stats(portvals_manual_in, "ManualStrategy In-Sample")
    print_stats(portvals_learner_in, "StrategyLearner In-Sample")
    print_stats(portvals_manual_out, "ManualStrategy Out-of-Sample")
    print_stats(portvals_learner_out, "StrategyLearner Out-of-Sample")


if __name__ == "__main__":
    run_experiment()
