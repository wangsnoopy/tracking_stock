import numpy as np

class BagLearner(object):
    def __init__(self, learner, kwargs={}, bags=20, boost=False, verbose=False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learners = []

    def author(self):
        return "awang758"

    def study_group(self):
        return "awang758"

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner using bootstrap aggregation
        """
        self.learners = []
        n_samples = data_x.shape[0]
        
        for i in range(self.bags):
            # Create new learner instance
            learner = self.learner(**self.kwargs)
            
            # Bootstrap sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            sample_x = data_x[indices]
            sample_y = data_y[indices]
            
            # Train learner on bootstrap sample
            learner.add_evidence(sample_x, sample_y)
            self.learners.append(learner)

    def query(self, points):
        """
        Estimate a set of test points given the model we built.
        """
        predictions = np.zeros((self.bags, points.shape[0]))
        
        # Get predictions from all learners
        for i, learner in enumerate(self.learners):
            predictions[i] = learner.query(points)
        
        # Return average of all predictions
        return np.mean(predictions, axis=0)