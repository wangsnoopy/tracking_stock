""""""

"""Assess a betting strategy.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
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
  		  	   		 	 	 			  		 			 	 	 		 		 	
Student Name: Tucker Balch (replace with your name)  		  	   		 	 	 			  		 			 	 	 		 		 	
GT User ID: tb34 (replace with your User ID)  		  	   		 	 	 			  		 			 	 	 		 		 	
GT ID: 900897987 (replace with your GT ID)  		  	   		 	 	 			  		 			 	 	 		 		 	
"""  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
import numpy as np  		  	   		 	 	 			  		 			 	 	 		 		 	
from matplotlib import pyplot as plt
  		  	   		 	 	 			  		 			 	 	 		 		 	
def author():  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    :return: The GT username of the student  		  	   		 	 	 			  		 			 	 	 		 		 	
    :rtype: str  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    return "awang758"  # replace tb34 with your Georgia Tech username.
  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
def gtid():  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    :return: The GT ID of the student  		  	   		 	 	 			  		 			 	 	 		 		 	
    :rtype: int  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    return 904081341  # replace with your GT ID number
  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
def get_spin_result(win_prob):  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param win_prob: The probability of winning  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type win_prob: float  		  	   		 	 	 			  		 			 	 	 		 		 	
    :return: The result of the spin.  		  	   		 	 	 			  		 			 	 	 		 		 	
    :rtype: bool  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    result = False  		  	   		 	 	 			  		 			 	 	 		 		 	
    if np.random.random() <= win_prob:  		  	   		 	 	 			  		 			 	 	 		 		 	
        result = True  		  	   		 	 	 			  		 			 	 	 		 		 	
    return result  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
def test_code():  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    Method to test your code  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    win_prob = 0.60  # set appropriately to the probability of a win  		  	   		 	 	 			  		 			 	 	 		 		 	
    np.random.seed(gtid())  # do this only once  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(get_spin_result(win_prob))  # test the roulette spin  		  	   		 	 	 			  		 			 	 	 		 		 	
    # add your code here to implement the experiments
    plot_f1()
    plot_f2()
    plot_f3()

def gambling_exp1(target=80, win_prob=18.0/38.0, max_spins=300):
    episode_winnings = 0
    winnings_record = []
    while episode_winnings < target and len(winnings_record) < max_spins:
        won = False
        bet_amount = 1
        while not won:
            spin_res = get_spin_result(win_prob)
            if spin_res:
                episode_winnings += bet_amount
                won = True
            else:
                episode_winnings -= bet_amount
                bet_amount *= 2

            winnings_record.append(episode_winnings)
            if episode_winnings >= target or len(winnings_record)>= 300:
                break
    return winnings_record

def run_episodes(num_episodes=1001, **kwargs):
    return [gambling_exp1(**kwargs) for _ in range(num_episodes)]

def plot_f1():
    plt.figure(figsize=(10, 6))
    for _ in range(10):
        winnings = gambling_exp1()
        plt.plot(winnings)
    plt.title("Figure 1:")
    plt.xlabel("Spin Number")
    plt.ylabel("Winnings")
    plt.ylim(-256, 100)
    plt.xlim(0, 300)
    plt.savefig("images/figure1.png")
    plt.show()

def plot_f2():
    data = np.array(1000)
    mean = np.mean(data, axis=0)
    std = np.std(data,axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(mean, lable="Mean", color="blue")
    plt.plot(mean + std, lable="Mean + Std", color="red")
    plt.plot(mean - std, lable="Mean - Std", color="green")

    plt.title("Figure 2:")
    plt.xlabel("Spin Number")
    plt.ylabel("Winnings")
    plt.ylim(-256, 100)
    plt.xlim(0, 300)
    plt.grid(True)
    plt.savefig("images/figure2.png")
    plt.show()

def plot_f3():
    data = np.array(1000)
    median = np.median(data, axis=0)
    std = np.std(data, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(median, lable="Median", color="orange")
    plt.plot(median + std, lable="Median + Std", linestyle="--", color="blue")
    plt.plot(median - std, lable="Median - Std", linestyle="--", color="green")

    plt.title("Figure 3:")
    plt.xlabel("Spin Number")
    plt.ylabel("Winnings")
    plt.ylim(-256, 100)
    plt.xlim(0, 300)
    plt.grid(True)
    plt.savefig("images/figure3.png")
    plt.show()



if __name__ == "__main__":  		  	   		 	 	 			  		 			 	 	 		 		 	
    test_code()  		  	   		 	 	 			  		 			 	 	 		 		 	
