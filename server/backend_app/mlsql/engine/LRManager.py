import numpy as np
import dill

class SimpleLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ (y - y_mean)
        self.intercept_ = y_mean - np.dot(self.coef_, X_mean)

    def predict(self, X):
        return X @ self.coef_ + self.intercept_

    def is_fitted(self):
        return self.coef_ is not None and self.intercept_ is not None


from .RegressionManager import RegressionManager
class LRManager(RegressionManager):
    def __init__(self):
        super().__init__(clonable=True)

    def create(self, name):
        estimator = SimpleLinearRegression()
        return self.save(name, estimator)

    def trainValidate(self, name, Xtrain, Xtest, yTrain, yTest):
        dic = {}
        self.train(name, Xtrain, yTrain)
        dic['training_mae'] = np.median(np.abs(yTrain - self.predict(name, Xtrain)))
        if Xtest is not None:
            dic['validation_mae'] = np.median(np.abs(yTest - self.predict(name, Xtest)))
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
        return estimator.predict(X)

    def evaluate(self, name, X, y):
        yPred = self.predict(name, X)
        dic = {}
        dic['training_mae'] = np.median(np.abs(y - yPred))
        return dic, yPred, y

    def isFitted(self, name):
        estimator = self.load(name)
        return estimator.is_fitted()

