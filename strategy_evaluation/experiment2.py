import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from marketsimcode import compute_portvals
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
from util import get_data
import os # Import os for directory creation/saving plots

# Remove trades_to_orders function, it's not needed with the updated marketsimcode.py
# def trades_to_orders(trades_df):
#     """
#     Convert trades DataFrame (daily holdings) to orders DataFrame.
#     """
#     # ... (remove this function entirely)

def normalize(df):
    return df / df.iloc[0]


def run():
    symbol = "JPM"
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    sv = 100000

    # Ensure image directory exists
    os.makedirs("images", exist_ok=True)

    # --- Part 1: Initial comparison (as you had it, but will be replaced by varying impact) ---
    # Manual Strategy setup - will keep this for initial comparison plot if needed, but primarily focus on SL
    manual = ManualStrategy()
    manual_trades = manual.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=sv)
    manual_portvals = compute_portvals(manual_trades, start_val=sv, commission=0.00, impact=0.0) # Set commission/impact as per experiment requirements

    # --- Part 2: Focus of Experiment 2: Varying Impact for StrategyLearner ---
    impact_values = [0.0, 0.005, 0.01] # Example impact values
    learner_portvals_by_impact = {}
    learner_trades_by_impact = {} # To check if trades differ

    for impact_val in impact_values:
        print(f"\nRunning StrategyLearner with impact = {impact_val}")
        sl = StrategyLearner(verbose=False, impact=impact_val, commission=0.0) # Commission is $0.00
        sl.add_evidence(symbol=symbol, sd=start_date, ed=end_date, sv=sv)
        learner_trades = sl.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=sv)
        learner_portvals = compute_portvals(learner_trades, start_val=sv, commission=0.00, impact=impact_val)

        learner_portvals_by_impact[impact_val] = normalize(learner_portvals)
        learner_trades_by_impact[impact_val] = learner_trades

        # Print stats for each impact level
        print_stats(normalize(learner_portvals), f"StrategyLearner (Impact={impact_val})")
        # Check if trades are different (you can add more robust checks)
        print(f"  Trades count: {np.sum(learner_trades.abs().values > 0)}")


    # Plotting for Experiment 2: Effect of Impact on StrategyLearner Performance
    plt.figure(figsize=(12, 7))
    plt.plot(normalize(manual_portvals), label="Manual Strategy (for comparison)", color='blue', linestyle='--') # Keep for context if desired
    for impact_val, portvals in learner_portvals_by_impact.items():
        plt.plot(portvals, label=f"Strategy Learner (Impact={impact_val})")

    plt.title(f"Experiment 2: Strategy Learner Performance vs. Market Impact ({symbol} In-Sample)")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/experiment2_impact_analysis.png")
    plt.close() # Change plt.show() to plt.close()

    # You might want another plot/table specifically showing a metric vs. impact
    # Example: Cumulative Return vs. Impact
    impact_cr = {imp: (pv.iloc[-1] / pv.iloc[0] - 1) for imp, pv in learner_portvals_by_impact.items()}
    
    impact_trade_counts = {imp: np.sum(trades.abs().values > 0) for imp, trades in learner_trades_by_impact.items()}

    print("\n--- Summary of Impact Effects on StrategyLearner (In-Sample) ---")
    print(f"{'Impact':<10} {'Cumulative Return':<20} {'Number of Trades':<20}")
    for imp in impact_values:
        cr_val = impact_cr.get(imp, float('nan'))
        trades_count = impact_trade_counts.get(imp, float('nan'))
        print(f"{imp:<10.4f} {cr_val:<20.6f} {trades_count:<20}")

    # Plot Cumulative Return vs Impact
    plt.figure(figsize=(8, 5))
    plt.plot(list(impact_cr.keys()), list(impact_cr.values()), marker='o')
    plt.title("Cumulative Return vs. Market Impact (Strategy Learner In-Sample)")
    plt.xlabel("Market Impact")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.savefig("images/experiment2_cr_vs_impact.png")
    plt.close()

    # Plot Number of Trades vs Impact
    plt.figure(figsize=(8, 5))
    plt.plot(list(impact_trade_counts.keys()), list(impact_trade_counts.values()), marker='o')
    plt.title("Number of Trades vs. Market Impact (Strategy Learner In-Sample)")
    plt.xlabel("Market Impact")
    plt.ylabel("Number of Trades")
    plt.grid(True)
    plt.savefig("images/experiment2_trades_vs_impact.png")
    plt.close()


    def print_stats(portvals, label):
        # Ensure portvals is a Series (extract if DataFrame)
        if isinstance(portvals, pd.DataFrame):
            portvals = portvals.iloc[:, 0]

        daily_returns = portvals.pct_change().dropna()
        cum_return = portvals.iloc[-1] / portvals.iloc[0] - 1
        std_daily_ret = daily_returns.std()
        mean_daily_ret = daily_returns.mean()

        print(f"{label}:")
        print(f"  Cumulative Return: {cum_return:.6f}") # Changed to .6f
        print(f"  Std of Daily Return: {std_daily_ret:.6f}") # Changed to .6f
        print(f"  Mean Daily Return: {mean_daily_ret:.6f}\n") # Changed to .6f


if __name__ == "__main__":
    run()
