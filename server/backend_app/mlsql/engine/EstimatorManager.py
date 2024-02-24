import os

import dill


class EstimatorManager:

    def __init__(self, clonable):
        self.clonable = clonable
        current_directory = os.path.dirname(__file__)
        self.estimator_folder =  os.path.join(current_directory, "estimators/")


    def getPath(self, estimatorName):
        return self.estimator_folder + estimatorName + ".dill"

    def save(self, name, estimator):
        fname = self.getPath(name)
        with open(fname, "wb") as fp:
            dill.dump(estimator, fp)

        return estimator

    def load(self, name):
        fname = self.getPath(name)
        with open(fname, "rb") as fp:
            estimator = dill.load(fp)

        return estimator