""""""  		  	   		 	 	 			  		 			 	 	 		 		 	
"""  		  	   		 	 	 			  		 			 	 	 		 		 	
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
    data = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1)
    np.random.shuffle(data)
    train_rows = int(0.6 * data.shape[0])
    train_x, train_y = data[:train_rows, :-1], data[:train_rows, -1]
    test_x, test_y = data[train_rows:, :-1], data[train_rows:, -1]
    
    # Baseline learners
    learners = [
        ("DTLearner", dt.DTLearner(leaf_size=1)),
        ("RTLearner", rt.RTLearner(leaf_size=1)),
        ("BagLearner", bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 1}, bags=20)),
        ("InsaneLearner", it.InsaneLearner())
    ]
    results = []
    for name, learner in learners:
        metrics = evaluate_learner(learner, train_x, train_y, test_x, test_y)
        results.append((name, *metrics))
    
    # Leaf size experiment
    leaf_sizes = [1, 5, 10, 20, 50]
    dt_rmse_train, dt_rmse_test, dt_times = [], [], []
    rt_rmse_train, rt_rmse_test, rt_times = [], [], []
    for ls in leaf_sizes:
        dt_metrics = evaluate_learner(dt.DTLearner(leaf_size=ls), train_x, train_y, test_x, test_y)
        rt_metrics = evaluate_learner(rt.RTLearner(leaf_size=ls), train_x, train_y, test_x, test_y)
        dt_rmse_train.append(dt_metrics[0]); dt_rmse_test.append(dt_metrics[2]); dt_times.append(dt_metrics[4])
        rt_rmse_train.append(rt_metrics[0]); rt_rmse_test.append(rt_metrics[2]); rt_times.append(rt_metrics[4])
    
    # Bag size experiment
    bag_sizes = [1, 5, 10, 20, 50]
    bag_rmse_train, bag_rmse_test, bag_times = [], [], []
    for bags in bag_sizes:
        metrics = evaluate_learner(bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 1}, bags=bags), train_x, train_y, test_x, test_y)
        bag_rmse_train.append(metrics[0]); bag_rmse_test.append(metrics[2]); bag_times.append(metrics[4])
    
    # Print results
    print("\nBaseline Learner Results:")
    for name, rmse_train, corr_train, rmse_test, corr_test, train_time in results:
        print(f"{name}:")
        print(f"  In-sample RMSE: {rmse_train:.4f}, Corr: {corr_train:.4f}")
        print(f"  Out-of-sample RMSE: {rmse_test:.4f}, Corr: {corr_test:.4f}")
        print(f"  Training time: {train_time:.4f} seconds")
    
    # Plot leaf size experiment
    plt.figure(figsize=(10, 5))
    plt.plot(leaf_sizes, dt_rmse_train, label='DTLearner In-sample')
    plt.plot(leaf_sizes, dt_rmse_test, label='DTLearner Out-of-sample')
    plt.plot(leaf_sizes, rt_rmse_train, label='RTLearner In-sample')
    plt.plot(leaf_sizes, rt_rmse_test, label='RTLearner Out-of-sample')
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.title('RMSE vs. Leaf Size')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(leaf_sizes, dt_times, label='DTLearner')
    plt.plot(leaf_sizes, rt_times, label='RTLearner')
    plt.xlabel('Leaf Size')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time vs. Leaf Size')
    plt.legend(); 
    plt.grid(True)
    plt.show()
    
    # Plot bag size experiment
    plt.figure(figsize=(10, 5))
    plt.plot(bag_sizes, bag_rmse_train, label='In-sample')
    plt.plot(bag_sizes, bag_rmse_test, label='Out-of-sample')
    plt.xlabel('Number of Bags')
    plt.ylabel('RMSE')
    plt.title('RMSE vs. Bag Size')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(bag_sizes, bag_times, label='BagLearner')
    plt.xlabel('Number of Bags')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time vs. Bag Size')
    plt.legend()
    plt.grid(True)
    plt.show()		  	   		 	 	 			  		 			 	 	 		 		 	
