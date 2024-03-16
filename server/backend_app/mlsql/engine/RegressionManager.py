from sklearn.linear_model import LinearRegression
from sklearn.metrics import median_absolute_error
import dill
from .EstimatorManager import EstimatorManager

class RegressionManager(EstimatorManager):

    def __init__(self, clonable):
        super(RegressionManager,self).__init__(clonable=clonable)

    def getPercentageError(self, y, yPred):
        return (y - yPred) * 100 / y
    def getAccuracy(self, y, yPred):
        # This method can be added for classification tasks
        correct_predictions = (y == yPred).sum()
        total_predictions = len(y)
        return (correct_predictions / total_predictions) * 100