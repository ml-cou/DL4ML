import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
class FormulaProcessor:

    def __init__(self, formula):

        self.formula = formula
        yX = self.stringFilter(re.split(r'\~+', self.formula))
        
        if len(yX) != 2:
            raise Exception(f"{formula} does not have $a~b$ format")

        self.fieldY = yX[0]
        self.formulaX = yX[1]
        self.fieldsX = self.stringFilter(re.split(r'\++', self.formulaX))

        pass
    

    def stringFilter(self, strings):

        filtered = []

        for s in strings:
            s = s.strip()
            if len(s) > 0:
                filtered.append(s)
        
        return filtered
    

    def getDataFromSQLDB(self, dataDb, trainingProfile, randomSeed = 42):
        """returns (XTrain, XValidation, yTrain, yValidation) training and validation X,y, does not support categorical data yet"""

        df = pd.read_sql(trainingProfile.source, dataDb.connection())

        X = df[self.fieldsX].values
        y = df[self.fieldY].values
        
        if trainingProfile.validationSplit <= 0:
            return X, None, y, None

        return train_test_split(X, y, test_size = trainingProfile.validationSplit , random_state = randomSeed)
    
    def getXFromSQL(self, dataDb, sql):
        """returns (XTest) training and validation X,y, does not support categorical data yet"""
        df = self.getDfFromSQL(dataDb, sql)
        return df[self.fieldsX].values

    def getDfFromSQL(self, dataDb, sql):
        """returns (XTest) training and validation X,y, does not support categorical data yet"""
        df = pd.read_sql(sql, dataDb.connection())
        return df
    
    def getDfAndXFromSQL(self, dataDb, sql, onlyPredictors=True):
        """returns (XTest) training and validation X,y, does not support categorical data yet"""
        df = pd.read_sql(sql, dataDb.connection())
        if onlyPredictors:
            return df[self.fieldsX], df[self.fieldsX].values
        else:
            return df, df[self.fieldsX].values
    
    def getYfromSQL(self, dataDb, sql):
        df = pd.read_sql(sql, dataDb.connection())
        return df [self.fieldY].values