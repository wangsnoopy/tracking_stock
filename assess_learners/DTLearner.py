import numpy as np

class DTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def author(self):
        return "awang758"

    def gtid(self):
        return 904081341

    def study_group(self):
        return "awang758"

    def add_evidence(self, data_x, data_y):
        self.tree = self.build_tree(data_x, data_y)
        if self.verbose:
            print(f"Tree shape: {self.tree.shape}")
            print(f"Tree:\n{self.tree}")
            print(f"Number of leaves: {np.sum(self.tree[:, 0] == -1)}")

    def query(self, points):
        pred = np.zeros(points.shape[0])
        max_depth = 1000  # Prevent infinite loops
        for i in range(points.shape[0]):
            node_idx = 0
            depth = 0
            while self.tree[node_idx, 0] != -1 and depth < max_depth:
                feature = int(self.tree[node_idx, 0])
                split_val = self.tree[node_idx, 1]
                if points[i, feature] <= split_val:
                    node_idx = int(self.tree[node_idx, 2])
                else:
                    node_idx = int(self.tree[node_idx, 3])
                depth += 1
            if depth >= max_depth:
                pred[i] = np.mean(self.tree[self.tree[:, 0] == -1, 1])  # Fallback
                if self.verbose:
                    print(f"Warning: Max depth reached for point {i}")
            else:
                pred[i] = self.tree[node_idx, 1]
        return pred

    def select_feature(self, X, Y):
        correlations = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            if np.std(X[:, i]) == 0:
                correlations[i] = 0
            else:
                corr = np.corrcoef(X[:, i], Y)[0, 1]
                correlations[i] = abs(corr) if not np.isnan(corr) else 0
        best_feature = np.argmax(correlations)
        split_val = np.median(X[:, best_feature])
        if self.verbose:
            print(f"Selected feature {best_feature} with correlation {correlations[best_feature]:.4f}, split at {split_val:.4f}")
        return best_feature, split_val

    def build_tree(self, X, Y):
        if X.shape[0] <= self.leaf_size or np.all(Y == Y[0]):
            leaf = np.array([[-1, np.mean(Y), np.nan, np.nan]])
            return leaf

        best_feature, split_val = self.select_feature(X, Y)
        left_mask = X[:, best_feature] <= split_val
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            leaf = np.array([[-1, np.mean(Y), np.nan, np.nan]])
            return leaf

        left_tree = self.build_tree(X[left_mask], Y[left_mask])
        right_tree = self.build_tree(X[right_mask], Y[right_mask])

        # Fix child indices
        root = np.array([[best_feature, split_val, 1, left_tree.shape[0] + 1]], dtype=np.float64)
        root[0, 2] = int(root[0, 2])  # Left child index
        root[0, 3] = int(root[0, 3])  # Right child index
        tree = np.vstack((root, left_tree, right_tree))
        return tree
    