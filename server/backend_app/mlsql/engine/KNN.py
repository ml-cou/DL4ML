import numpy as np
from collections import Counter

class SimpleKNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Calculate the distances between x and all examples in the training set
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        # Get the k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def is_fitted(self):
        return self.X_train is not None and self.y_train is not None

from .RegressionManager import RegressionManager


class KNNManager(RegressionManager):
    def __init__(self):
        super().__init__(clonable=True)

    def create(self, name, k=3):
        estimator = SimpleKNNClassifier(k=k)
        return self.save(name, estimator)

    def trainValidate(self, name, Xtrain, Xtest, yTrain, yTest):
        dic = {}
        self.train(name, Xtrain, yTrain)
        dic['training_accuracy'] = np.mean(yTrain == self.predict(name, Xtrain))
        if Xtest is not None:
            dic['validation_accuracy'] = np.mean(yTest == self.predict(name, Xtest))
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
        dic['accuracy'] = np.mean(y == yPred)
        return dic, yPred

    def isFitted(self, name):
        estimator = self.load(name)
        return estimator.is_fitted()
