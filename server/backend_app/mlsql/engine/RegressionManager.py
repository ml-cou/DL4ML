from sklearn.linear_model import LinearRegression
from sklearn.metrics import median_absolute_error
import dill
from .EstimatorManager import EstimatorManager

class RegressionManager(EstimatorManager):

    def __init__(self, clonable):
        super(RegressionManager,self).__init__(clonable=clonable)

    def getPercentageError(self, y, yPred):
        return (y - yPred) * 100 / y
