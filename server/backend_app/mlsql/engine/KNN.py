from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import median_absolute_error
import dill
from .RegressionManager import RegressionManager


class KNNManager(RegressionManager):

    def __init__(self):
        super(KNNManager, self).__init__(clonable=True)

    def create(self, name):
        estimator = KNeighborsRegressor()
        return self.save(name, estimator)

    def trainValidate(self, name, Xtrain, Xtest, yTrain, yTest):

        dic = {}
        self.train(name, Xtrain, yTrain)
        dic['training_mae'] = median_absolute_error(yTrain, self.predict(name, Xtrain))
        if Xtest is not None:
            dic['validation_mae'] = median_absolute_error(yTest, self.predict(name, Xtest))
        return dic

    def train(self, name, X, y):
        print(f"Starting training for model: {name}")
        estimator = self.load(name)
        try:
            estimator.fit(X, y)
            print(f"Model {name} trained successfully.")
            return self.save(name, estimator)
        except Exception as e:
            print(f"Error during training of model {name}: {e}")
            raise e

    def predict(self, name, X):
        estimator = self.load(name)
        pred = estimator.predict(X)
        return pred

    def evaluate(self, name, X, y):
        yPred = self.predict(name, X)
        dic = {}
        dic['training_mae'] = median_absolute_error(y, yPred)
        return dic, yPred

    def isFitted(self, name):
        # Example implementation; adjust according to your model management system
        estimator = self.load(name)
        return hasattr(estimator, 'n_neighbors')  # Check if the model has 'n_neighbors' attribute
