""""""
"""

error:
Traceback (most recent call last):
  File "/Users/awang/Documents/ML4T_2025Summer/assess_learners/testlearner.py", line 73, in <module>
    corr_test.append(metrics[5])
IndexError: tuple index out of range

  		  	   		 	 	 			  		 			 	 	 		 		 	
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
  		  	   		 	 	 			  		 			 	 	 		 		 	
import numpy as np
import sys
import math
import time
import matplotlib.pyplot as plt
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it 		  	   		 	 	 			  		 			 	 	 		 		 	

def evaluate_learner(learner, train_x, train_y, test_x, test_y):
    start = time.time()
    learner.add_evidence(train_x, train_y)
    train_time = time.time() - start
    pred_y_train = learner.query(train_x)
    rmse_train = math.sqrt(((train_y - pred_y_train) ** 2).mean())
    corr_train = np.corrcoef(pred_y_train, train_y)[0, 1]
    pred_y_test = learner.query(test_x)
    rmse_test = math.sqrt(((test_y - pred_y_test) ** 2).mean())
    corr_test = np.corrcoef(pred_y_test, test_y)[0, 1]
    return rmse_train, corr_train, rmse_test, corr_test, train_time
 	   		 	 	 			  		 			 	 	 		 		 	
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)

    # Read and split data
    np.random.seed(904081341)  # Replace with your GT ID
    data = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1)[:, 1:]  # Skip date column
    np.random.shuffle(data)
    train_rows = int(0.6 * data.shape[0])
    train_x, train_y = data[:train_rows, :-1], data[:train_rows, -1]
    test_x, test_y = data[train_rows:, :-1], data[train_rows:, -1]
    
    # Experiment 1: Overfitting with DTLearner
    leaf_sizes = list(range(1, 21))
    rmse_train, rmse_test, corr_train, corr_test = [], [], [], []
    with open("p3_results.txt", "w") as f:
        f.write("Experiment 1: DTLearner Overfitting\n")
        f.write("Leaf Size, In-sample RMSE, Out-of-sample RMSE, In-sample Corr, Out-of-sample Corr\n")
        for ls in leaf_sizes:
            learner = dt.DTLearner(leaf_size=ls, verbose=True)
            metrics = evaluate_learner(learner, train_x, train_y, test_x, test_y)
            rmse_train.append(metrics[0])
            rmse_test.append(metrics[1])
            corr_train.append(metrics[4])
            corr_test.append(metrics[5])
            f.write(f"{ls},{metrics[0]:.4f},{metrics[1]:.4f},{metrics[4]:.4f},{metrics[5]:.4f}\n")
            if ls == 1:
                print(f"DTLearner (leaf_size=1): In-sample Corr: {metrics[4]:.4f}, Out-of-sample Corr: {metrics[5]:.4f}")
            if ls == 50 or ls == max(leaf_sizes):
                print(f"DTLearner (leaf_size={ls}): In-sample Corr: {metrics[4]:.4f}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(leaf_sizes, rmse_train, label='In-sample RMSE')
    plt.plot(leaf_sizes, rmse_test, label='Out-of-sample RMSE')
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.title('Experiment 1: DTLearner RMSE vs. Leaf Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiment_1_rmse.png')
    plt.close()
    
    # Experiment 3: DTLearner vs. RTLearner
    num_trials = 10
    leaf_size = 30
    dt_mae, rt_mae, dt_times, rt_times = [], [], [], []
    with open("p3_results.txt", "a") as f:
        f.write("\nExperiment 3: DTLearner vs. RTLearner\n")
        f.write("Trial, DTLearner MAE, RTLearner MAE, DTLearner Time, RTLearner Time\n")
        for i in range(num_trials):
            # Reshuffle data for each trial
            np.random.shuffle(data)
            train_x, train_y = data[:train_rows, :-1], data[:train_rows, -1]
            test_x, test_y = data[train_rows:, :-1], data[train_rows:, -1]
            
            dt_learner = dt.DTLearner(leaf_size=leaf_size, verbose=True)
            rt_learner = rt.RTLearner(leaf_size=leaf_size, verbose=True)
            dt_metrics = evaluate_learner(dt_learner, train_x, train_y, test_x, test_y)
            rt_metrics = evaluate_learner(rt_learner, train_x, train_y, test_x, test_y)
            
            dt_mae.append(dt_metrics[3])
            rt_mae.append(rt_metrics[3])
            dt_times.append(dt_metrics[6])
            rt_times.append(rt_metrics[6])
            f.write(f"{i+1},{dt_metrics[3]:.4f},{rt_metrics[3]:.4f},{dt_metrics[6]:.4f},{rt_metrics[6]:.4f}\n")
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_trials+1), dt_mae, label='DTLearner MAE')
    plt.plot(range(1, num_trials+1), rt_mae, label='RTLearner MAE')
    plt.xlabel('Trial')
    plt.ylabel('Mean Absolute Error')
    plt.title('Experiment 3: DTLearner vs. RTLearner MAE')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiment_3_mae.png')
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_trials+1), dt_times, label='DTLearner Training Time')
    plt.plot(range(1, num_trials+1), rt_times, label='RTLearner Training Time')
    plt.xlabel('Trial')
    plt.ylabel('Training Time (seconds)')
    plt.title('Experiment 3: DTLearner vs. RTLearner Training Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiment_3_time.png')
    plt.close()