import numpy as np

class RTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

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
    
    def select_feature(self, X, Y):
        best_feature = np.random.randint(0, X.shape[1])
        split_val = np.median(X[:, best_feature])
        return best_feature, split_val

    def build_tree(self, data, start_idx):
        X = data[:, :-1]
        Y = data[:, -1]
        if data.shape[0] <= self.leaf_size or np.all(Y == Y[0]):
            return np.array([[-1, np.mean(Y), -1, -1]]), start_idx
        best_feature, split_val = self.select_feature(X, Y)
        left_mask = X[:, best_feature] <= split_val
        right_mask = ~left_mask
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return np.array([[-1, np.mean(Y), -1, -1]]), start_idx
        left_tree, left_idx = self.build_tree(data[left_mask], start_idx + 1)
        right_tree, right_idx = self.build_tree(data[right_mask], left_idx)
        node = np.array([[best_feature, split_val, start_idx + 1, left_idx]])
        tree = np.vstack([node, left_tree, right_tree])
        return tree, right_idx

    def add_evidence(self, data_x, data_y):
        data_y = data_y.reshape(-1, 1)
        data = np.hstack([data_x, data_y])
        self.tree, _ = self.build_tree(data, 0)
        if self.verbose:
            print("Tree shape:", self.tree.shape)
            print(self.tree)

    def query(self, points):
        Ypred = np.zeros(points.shape[0])
        for i, point in enumerate(points):
            node_idx = 0
            while self.tree[node_idx, 0] != -1:
                feature = int(self.tree[node_idx, 0])
                split_val = self.tree[node_idx, 1]
                node_idx = int(self.tree[node_idx, 2]) if point[feature] <= split_val else int(self.tree[node_idx, 3])
            Ypred[i] = self.tree[node_idx, 1]
        return Ypred