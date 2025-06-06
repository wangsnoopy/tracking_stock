import numpy as np

class BagLearner(object):
    def __init__(self, learner, kwargs, bags=20, boost=False, verbose=False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learners = [self.learner(**kwargs) for i in range(bags)]

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
                                                                                            
    def study_group():
        return "awang758"
    
    def add_evidence(self, data_x, data_y):
        n_samples = data_x.shape[0]
        for learner in self.learners:
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample = data_x[indices]
            Y_sample = data_y[indices]
            learner.add_evidence(X_sample, Y_sample)
        if self.verbose:
            print(f"Trained {self.bags} learners with bootstrap sampling")

    def query(self, points):
        predictions = np.array([learner.query(points) for learner in self.learners])
        Ypred = np.mean(predictions, axis=0)
        return Ypred