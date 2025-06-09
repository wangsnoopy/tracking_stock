""""""
""" 			  		 			 	 	 		 		 	
terminal:

Test a learner.  (c) 2015 Tucker Balch  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			 	 	 		 		 	
or edited.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		 	 	 			  		 			 	 	 		 		 	
"""  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import LinRegLearner as lrl
import DTLearner as dt
import BagLearner as bl
import RTLearner as rt
import InsaneLearner as it
import time


def convert_to_float(value):
    try:
        return float(value)
    except ValueError:
        return "no"

# Add default argument for easier running
# sys.argv.append("Data/Istanbul.csv")

if __name__ == "__main__":
    np.random.seed(904081341)

    if len(sys.argv) == 1:
        sys.argv.append("Data/Istanbul.csv")

    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)

    with open(sys.argv[1], "r") as file:
        raw_lines = [line.strip().split(",") for line in file.readlines()]

    # Check if first row contains strings
    first_row_check = ['no'] * len(raw_lines[0])
    converted_first_row = [convert_to_float(val) for val in raw_lines[0]]
    if first_row_check == converted_first_row:
        del raw_lines[0]

    # Remove string-type columns if present
    for row in raw_lines:
        row[:] = [val for val in row if convert_to_float(val) != "no"]

    # Convert to NumPy array
    data = np.array(raw_lines, dtype=float)

    # Split data
    total_rows = data.shape[0]
    train_size = int(0.6 * total_rows)
    x_train = data[:train_size, :-1]
    y_train = data[:train_size, -1]
    x_test = data[train_size:, :-1]
    y_test = data[train_size:, -1]

    # ---------- DT Experiments ----------
    dt_rmse_list = []
    for leaf in range(1, 51):
        model_dt = dt.DTLearner(leaf_size=leaf, verbose=False)
        model_dt.add_evidence(x_train, y_train)

        pred_train = model_dt.query(x_train)
        rmse_train = math.sqrt(((y_train - pred_train) ** 2).mean())

        pred_test = model_dt.query(x_test)
        rmse_test = math.sqrt(((y_test - pred_test) ** 2).mean())

        dt_rmse_list.append([rmse_train, rmse_test])

    df_dt = pd.DataFrame(dt_rmse_list, index=range(1, 51), columns=["training data", "testing data"])
    ax = df_dt.plot(title="Effect of leaf size on dt learner")
    ax.set_xlabel("Leaf size")
    ax.set_ylabel("Root mean square error")
    plt.savefig("Figure1.png")
    plt.close()

    # ---------- BagLearner Experiments ----------
    bag_count = 100
    bag_rmse_list = []
    for leaf in range(1, 51):
        model_bag = bl.BagLearner(dt.DTLearner, bags=bag_count, kwargs={'leaf_size': leaf}, boost=True, verbose=False)
        model_bag.add_evidence(x_train, y_train)

        pred_train = model_bag.query(x_train)
        rmse_train = math.sqrt(((y_train - pred_train) ** 2).mean())

        pred_test = model_bag.query(x_test)
        rmse_test = math.sqrt(((y_test - pred_test) ** 2).mean())

        bag_rmse_list.append([rmse_train, rmse_test])

    df_bag = pd.DataFrame(bag_rmse_list, index=range(1, 51), columns=["training data", "testing data"])
    ax = df_bag.plot(title=f"{bag_count}-bag case on dt learner")
    ax.set_xlabel("Leaf size")
    ax.set_ylabel("Root mean square error")
    plt.savefig("Figure2.png")
    plt.close()

    # ---------- RT vs DT Training Time and MAD ----------
    train_time = []
    mad_values = []

    for leaf in range(1, 51):
        # DT training
        t0 = time.time()
        dt_model = dt.DTLearner(leaf_size=leaf, verbose=False)
        dt_model.add_evidence(x_train, y_train)
        t1 = time.time()
        dt_time = t1 - t0

        dt_preds = dt_model.query(x_test)
        dt_mad = np.mean(np.abs(dt_preds - y_test))

        # RT training
        t2 = time.time()
        rt_model = rt.RTLearner(leaf_size=leaf, verbose=False)
        rt_model.add_evidence(x_train, y_train)
        t3 = time.time()
        rt_time = t3 - t2

        rt_preds = rt_model.query(x_test)
        rt_mad = np.mean(np.abs(rt_preds - y_test))

        train_time.append([dt_time, rt_time])
        mad_values.append([dt_mad, rt_mad])

    df_time = pd.DataFrame(np.array(train_time), index=range(1, 51), columns=["DT training", "RT training"])
    ax = df_time.plot(title="Comparison of training time between DT and RT")
    ax.set_xlabel("Leaf size")
    ax.set_ylabel("Training time (s)")
    plt.savefig("Figure3-1.png")
    plt.close()

    df_mad = pd.DataFrame(np.array(mad_values), index=range(1, 51), columns=["DT testing", "RT testing"])
    ax = df_mad.plot(title="Comparison of MAD between DT and RT")
    ax.set_xlabel("Leaf size")
    ax.set_ylabel("Mean absolute deviation")
    plt.yticks(np.arange(0.003, 0.007, step=0.001))
    plt.savefig("Figure3-2.png")
    plt.close()
