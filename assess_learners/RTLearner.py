import numpy as np

class RTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def author(self):
        return "awang758"

    def study_group(self):
        return "awang758"

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner
        """
        self.tree = self.build_tree(data_x, data_y)

    def query(self, points):
        """
        Estimate a set of test points given the model we built.
        """
        pred = np.zeros(points.shape[0])
        for i in range(points.shape[0]):
            node_idx = 0
            while self.tree[node_idx, 0] != -1:  # Not a leaf
                feature = int(self.tree[node_idx, 0])
                split_val = self.tree[node_idx, 1]
                if points[i, feature] <= split_val:
                    node_idx = int(self.tree[node_idx, 2])  # Left child
                else:
                    node_idx = int(self.tree[node_idx, 3])  # Right child
            pred[i] = self.tree[node_idx, 1]  # Leaf value
        return pred

    def build_tree(self, data_x, data_y):
        """
        Build random tree recursively
        """
        # Base case: create leaf if conditions met
        if data_x.shape[0] <= self.leaf_size or np.all(data_y == data_y[0]):
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])

        # Randomly select feature to split on
        best_feature = np.random.randint(0, data_x.shape[1])
        split_val = np.median(data_x[:, best_feature])
        
        # Split data
        left_mask = data_x[:, best_feature] <= split_val
        right_mask = ~left_mask
        
        # If split doesn't separate data, make leaf
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])
        
        # Recursively build subtrees
        left_tree = self.build_tree(data_x[left_mask], data_y[left_mask])
        right_tree = self.build_tree(data_x[right_mask], data_y[right_mask])
        
        # Create root node
        root = np.array([[best_feature, split_val, 1, 1 + left_tree.shape[0]]])
        
        # Combine root with subtrees
        tree = np.vstack((root, left_tree, right_tree))
        return tree