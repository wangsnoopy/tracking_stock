# testproject.py
import datetime as dt
import matplotlib.pyplot as plt

from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
import experiment1
import experiment2
import pandas as pd
import os

def run_manual_strategy():
    print("Running Manual Strategy...")
    ms = ManualStrategy()
    ms.add_evidence(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    manual_trades = ms.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    print("Manual strategy trades:")
    print(manual_trades.head())

def run_strategy_learner():
    print("Running Strategy Learner...")
    sl = StrategyLearner(verbose=False, impact=0.0)
    sl.add_evidence(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    learner_trades = sl.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    print("Strategy Learner trades:")
    print(learner_trades.head())

def run_experiment1():
    print("Running Experiment 1 (Impact Analysis)...")
    experiment1.run()

def run_experiment2():
    print("Running Experiment 2 (Insample vs Outsample Performance)...")
    experiment2.run()

def main():
    # Create output directory if needed
    os.makedirs("images", exist_ok=True)
    
    run_manual_strategy()
    run_strategy_learner()
    run_experiment1()
    run_experiment2()

if __name__ == "__main__":
    main()
