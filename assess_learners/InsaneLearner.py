import numpy as np
import BagLearner as bl
import LinRegLearner as lrl

class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.learnerlist = [bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=verbose) for _ in range(20)]
    def author(self):
        return "awang758"
    def add_evidence(self, data_x, data_y):
        [learner.add_evidence(data_x, data_y) for learner in self.learnerlist]
    def query(self, points):
        predictions = np.array([learner.query(points) for learner in self.learnerlist])
        return predictions.mean(axis=0)
