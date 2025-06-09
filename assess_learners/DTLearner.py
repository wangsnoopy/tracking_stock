import numpy as np

class DTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def author(self):
        return "awang758"

    def study_group(self):
        return "awang758"

    def best_feature(self, data_x, data_y):
        max_corr = 0
        best_idx = 0
        for i in range(data_x.shape[1]):
            if np.std(data_x[:, i]) == 0 or np.std(data_y) == 0:
                continue
            corr_matrix = np.corrcoef(data_x[:, i], data_y)
            corr = abs(corr_matrix[0, 1])
            if corr >= max_corr:
                max_corr = corr
                best_idx = i
        return best_idx

    def build_tree(self, data):
        if data.shape[0] <= self.leaf_size or np.std(data[:, -1]) == 0:
            leaf_node = np.full((1, 4), -100.0)
            leaf_node[0, 0] = -1
            leaf_node[0, 1] = np.mean(data[:, -1])
            return leaf_node

        feature_idx = self.best_feature(data[:, :-1], data[:, -1])
        split_val = np.median(data[:, feature_idx])

        if np.all(data[:, feature_idx] <= split_val):
            return np.array([[-1, np.mean(data[:, -1]), -100, -100]])

        #recursion starts here:
        left_tree = self.build_tree(data[data[:, feature_idx] <= split_val])
        right_tree = self.build_tree(data[data[:, feature_idx] > split_val])
        root = np.array([[feature_idx, split_val, 1, left_tree.shape[0] + 1]])
        return np.vstack((root, left_tree, right_tree))

    def add_evidence(self, data_x, data_y):
        data = np.column_stack([data_x, data_y]) #column stack is used because concatenate or append somehow doesn't work
        self.tree = self.build_tree(data)
        if self.verbose:
            print("tree:\n", self.tree)
            print("tree shape:\n", self.tree.shape)


        # build and save the model

    def query(self, points):
        tree = self.tree
        results = np.array(())
        for point in points:
            node = 0
            while int(tree[node, 0]) != -1:
                feature_idx = int(tree[node, 0])
                if point[feature_idx] <= tree[node, 1]:
                    node += int(tree[node, 2])
                else:
                    node += int(tree[node, 3])
            # thisy = tree[node, 1]
            # thisy = np.array((thisy))
            # testy = np.append(testy, thisy)
            prediction = self.tree[node, 1]
            results = np.append(results, prediction)
        return results