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
  		  	   		 	 	 			  		 			 	 	 		 		 	
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import math

import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as il

def evaluate_learner(learner, train_x, train_y, test_x, test_y):
    """
    Evaluate a learner and return various performance metrics
    """
    start_time = time.time()
    learner.add_evidence(train_x, train_y)
    train_time = time.time() - start_time
    
    # Get predictions
    pred_y_train = learner.query(train_x)
    pred_y_test = learner.query(test_x)
    
    # Calculate metrics
    rmse_train = math.sqrt(((train_y - pred_y_train) ** 2).mean())
    rmse_test = math.sqrt(((test_y - pred_y_test) ** 2).mean())
    mae_train = np.abs(train_y - pred_y_train).mean()
    mae_test = np.abs(test_y - pred_y_test).mean()
    
    # Calculate correlations
    corr_train = 0.0
    corr_test = 0.0
    if np.std(pred_y_train) > 0 and np.std(train_y) > 0:
        corr_train = np.corrcoef(pred_y_train, train_y)[0, 1]
        if np.isnan(corr_train):
            corr_train = 0.0
    
    if np.std(pred_y_test) > 0 and np.std(test_y) > 0:
        corr_test = np.corrcoef(pred_y_test, test_y)[0, 1]
        if np.isnan(corr_test):
            corr_test = 0.0
    
    return rmse_train, rmse_test, mae_train, mae_test, corr_train, corr_test, train_time

def experiment1_overfitting(train_x, train_y, test_x, test_y):
    """
    Experiment 1: Study overfitting with DTLearner
    """
    print("Running Experiment 1: DTLearner Overfitting Analysis")
    
    leaf_sizes = [1, 5, 10, 20, 50]
    rmse_train_list = []
    rmse_test_list = []
    corr_train_list = []
    corr_test_list = []
    
    with open("p3_results.txt", "w") as f:
        f.write("Experiment 1: DTLearner Overfitting Analysis\n")
        f.write("Leaf Size, In-sample RMSE, Out-of-sample RMSE, In-sample Corr, Out-of-sample Corr\n")
        
        for leaf_size in leaf_sizes:
            learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
            rmse_train, rmse_test, mae_train, mae_test, corr_train, corr_test, train_time = evaluate_learner(
                learner, train_x, train_y, test_x, test_y)
            
            rmse_train_list.append(rmse_train)
            rmse_test_list.append(rmse_test)
            corr_train_list.append(corr_train)
            corr_test_list.append(corr_test)
            
            f.write(f"{leaf_size},{rmse_train:.6f},{rmse_test:.6f},{corr_train:.6f},{corr_test:.6f}\n")
            print(f"Leaf size {leaf_size}: In-sample RMSE={rmse_train:.4f}, Out-sample RMSE={rmse_test:.4f}")
    
    # Create plots
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(leaf_sizes, rmse_train_list, 'b-o', label='In-sample RMSE')
    plt.plot(leaf_sizes, rmse_test_list, 'r-o', label='Out-of-sample RMSE')
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.title('Experiment 1: DTLearner RMSE vs Leaf Size')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(leaf_sizes, corr_train_list, 'b-o', label='In-sample Correlation')
    plt.plot(leaf_sizes, corr_test_list, 'r-o', label='Out-of-sample Correlation')
    plt.xlabel('Leaf Size')
    plt.ylabel('Correlation')
    plt.title('Experiment 1: DTLearner Correlation vs Leaf Size')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('experiment1_dtlearner_overfitting.png', dpi=150, bbox_inches='tight')
    plt.close()

def experiment2_bagging(train_x, train_y, test_x, test_y):
    """
    Experiment 2: Study bagging effects with different leaf sizes
    """
    print("Running Experiment 2: Bagging Analysis")
    
    leaf_sizes = [1, 5, 10, 20, 50]
    rmse_train_dt = []
    rmse_test_dt = []
    rmse_train_bag = []
    rmse_test_bag = []
    
    with open("p3_results.txt", "a") as f:
        f.write("\nExperiment 2: Bagging Analysis\n")
        f.write("Leaf Size, DT Train RMSE, DT Test RMSE, Bag Train RMSE, Bag Test RMSE\n")
        
        for leaf_size in leaf_sizes:
            # Single DTLearner
            dt_learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
            dt_metrics = evaluate_learner(dt_learner, train_x, train_y, test_x, test_y)
            
            # BagLearner with DTLearners
            bag_learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": leaf_size}, 
                                      bags=20, boost=False, verbose=False)
            bag_metrics = evaluate_learner(bag_learner, train_x, train_y, test_x, test_y)
            
            rmse_train_dt.append(dt_metrics[0])
            rmse_test_dt.append(dt_metrics[1])
            rmse_train_bag.append(bag_metrics[0])
            rmse_test_bag.append(bag_metrics[1])
            
            f.write(f"{leaf_size},{dt_metrics[0]:.6f},{dt_metrics[1]:.6f},{bag_metrics[0]:.6f},{bag_metrics[1]:.6f}\n")
            print(f"Leaf size {leaf_size}: DT Test RMSE={dt_metrics[1]:.4f}, Bag Test RMSE={bag_metrics[1]:.4f}")
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(leaf_sizes, rmse_test_dt, 'b-o', label='DTLearner (Out-of-sample)')
    plt.plot(leaf_sizes, rmse_test_bag, 'r-o', label='BagLearner (Out-of-sample)')
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.title('Experiment 2: DTLearner vs BagLearner Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiment2_bagging_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def experiment3_dt_vs_rt(data):
    """
    Experiment 3: Compare DTLearner vs RTLearner with multiple random runs
    """
    print("Running Experiment 3: DTLearner vs RTLearner Comparison")
    
    num_trials = 10
    leaf_size = 1
    train_rows = int(0.6 * data.shape[0])
    
    dt_mae_list = []
    rt_mae_list = []
    dt_time_list = []
    rt_time_list = []
    
    with open("p3_results.txt", "a") as f:
        f.write("\nExperiment 3: DTLearner vs RTLearner Comparison\n")
        f.write("Trial, DT MAE, RT MAE, DT Time, RT Time\n")
        
        for trial in range(num_trials):
            # Shuffle data for each trial
            np.random.shuffle(data)
            train_x = data[:train_rows, :-1]
            train_y = data[:train_rows, -1]
            test_x = data[train_rows:, :-1]
            test_y = data[train_rows:, -1]
            
            # Test DTLearner
            dt_learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
            dt_metrics = evaluate_learner(dt_learner, train_x, train_y, test_x, test_y)
            
            # Test RTLearner
            rt_learner = rt.RTLearner(leaf_size=leaf_size, verbose=False)
            rt_metrics = evaluate_learner(rt_learner, train_x, train_y, test_x, test_y)
            
            dt_mae_list.append(dt_metrics[3])
            rt_mae_list.append(rt_metrics[3])
            dt_time_list.append(dt_metrics[6])
            rt_time_list.append(rt_metrics[6])
            
            f.write(f"{trial+1},{dt_metrics[3]:.6f},{rt_metrics[3]:.6f},{dt_metrics[6]:.6f},{rt_metrics[6]:.6f}\n")
            print(f"Trial {trial+1}: DT MAE={dt_metrics[3]:.4f}, RT MAE={rt_metrics[3]:.4f}")
    
    # Create plots
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    trials = list(range(1, num_trials+1))
    plt.plot(trials, dt_mae_list, 'b-o', label='DTLearner MAE')
    plt.plot(trials, rt_mae_list, 'r-o', label='RTLearner MAE')
    plt.xlabel('Trial')
    plt.ylabel('Mean Absolute Error')
    plt.title('Experiment 3: DTLearner vs RTLearner MAE')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(trials, dt_time_list, 'b-o', label='DTLearner Time')
    plt.plot(trials, rt_time_list, 'r-o', label='RTLearner Time')
    plt.xlabel('Trial')
    plt.ylabel('Training Time (seconds)')
    plt.title('Experiment 3: Training Time Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('experiment3_dt_vs_rt_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    with open("p3_results.txt", "a") as f:
        f.write(f"\nExperiment 3 Summary:\n")
        f.write(f"DTLearner - Mean MAE: {np.mean(dt_mae_list):.6f}, Std MAE: {np.std(dt_mae_list):.6f}\n")
        f.write(f"RTLearner - Mean MAE: {np.mean(rt_mae_list):.6f}, Std MAE: {np.std(rt_mae_list):.6f}\n")
        f.write(f"DTLearner - Mean Time: {np.mean(dt_time_list):.6f}, Std Time: {np.std(dt_time_list):.6f}\n")
        f.write(f"RTLearner - Mean Time: {np.mean(rt_time_list):.6f}, Std Time: {np.std(rt_time_list):.6f}\n")

def test_insane_learner(train_x, train_y, test_x, test_y):
    """
    Test InsaneLearner
    """
    print("Testing InsaneLearner...")
    
    learner = il.InsaneLearner(verbose=False)
    metrics = evaluate_learner(learner, train_x, train_y, test_x, test_y)
    
    with open("p3_results.txt", "a") as f:
        f.write(f"\nInsaneLearner Results:\n")
        f.write(f"In-sample RMSE: {metrics[0]:.6f}\n")
        f.write(f"Out-of-sample RMSE: {metrics[1]:.6f}\n")
        f.write(f"In-sample Correlation: {metrics[4]:.6f}\n")
        f.write(f"Out-of-sample Correlation: {metrics[5]:.6f}\n")
        f.write(f"Training Time: {metrics[6]:.6f} seconds\n")
    
    print(f"InsaneLearner - Out-of-sample RMSE: {metrics[1]:.4f}, Training Time: {metrics[6]:.2f}s")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    
    print("Starting testlearner.py experiments...")
    start_time = time.time()
    
    # Set random seed for reproducibility
    np.random.seed(123456789)
    
    # Read data
    try:
        data = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1)
        # Skip the first column (date) if it exists
        if data.shape[1] > 2:  # More than just X and Y
            data = data[:, 1:]  # Skip first column (date)
        print(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Shuffle and split data
    np.random.shuffle(data)
    train_rows = int(0.6 * data.shape[0])
    
    train_x = data[:train_rows, :-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, :-1]
    test_y = data[train_rows:, -1]
    
    print(f"Training set: {train_x.shape[0]} samples, {train_x.shape[1]} features")
    print(f"Test set: {test_x.shape[0]} samples, {test_x.shape[1]} features")
    
    # Run experiments
    try:
        experiment1_overfitting(train_x, train_y, test_x, test_y)
        experiment2_bagging(train_x, train_y, test_x, test_y)
        experiment3_dt_vs_rt(data)  # Pass original data for reshuffling
        test_insane_learner(train_x, train_y, test_x, test_y)
        
        total_time = time.time() - start_time
        print(f"\nAll experiments completed successfully in {total_time:.2f} seconds")
        
        with open("p3_results.txt", "a") as f:
            f.write(f"\nTotal execution time: {total_time:.2f} seconds\n")
            
    except Exception as e:
        print(f"Error during experiments: {e}")
        import traceback
        traceback.print_exc()