import numpy as np
import random

class BagLearner(object):
    def __init__(self, learner, kwargs, bags = 10, boost = True, verbose = True):
        self.verbose = verbose
        self.boost = boost
        self.bags = bags
        self.learner = learner
        self.kwargs = kwargs
        self.baglist = []

    def author(self):
        return "awang758"

    def study_group(self):
        return "awang758"

    def add_evidence(self, data_x, data_y):
        num_samples = data_x.shape[0]

        for _ in range(self.bags):
            sample_indices = np.random.choice(num_samples, num_samples, replace=True)
            bag_x = data_x[sample_indices]
            bag_y = data_y[sample_indices]

            learner_instance = self.learner(**self.kwargs)
            learner_instance.add_evidence(bag_x, bag_y)
            self.baglist.append(learner_instance)

        if self.verbose:
            print("bag number:\n", self.bags)

    def query(self, points):
        predictions = np.array([learner.query(points) for learner in self.baglist])
        return predictions.mean(axis=0)