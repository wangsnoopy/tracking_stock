import numpy as np
import BagLearner as bl
import LinRegLearner as lrl

class BagLearner(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.learners=[bl.BagLearner(learner=lrl.LinRegLearner,kwargs={},bags=20,boost=False,verbose=verbose) for i in range(20)]
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
        for learner in self.learners:
            learner.add_evidence(data_x, data_y)
        if self.verbose:
            print("Trained 20 BagLearners")

    def query(self, points):
        predictions = [learner.query(points) for learner in self.learners]
        return np.mean(predictions, axis=0)