import threading

import matplotlib
from peewee import *
from ..data.db import db
import datetime
from ..data.EstimatorMeta import EstimatorMeta
from ..data.TrainingProfile import TrainingProfile
from ..engine.LRManager import LRManager
from ..engine.KNN import KNNManager

from ..engine.FormulaProcessor import FormulaProcessor
import pprint
from matplotlib import pyplot as plt
import base64
import os
import pandas as pd
from io import BytesIO
import threading
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

class ASTProcessor:
    def __init__(self):
        self.pp = pprint.PrettyPrinter(indent=3)
        pass

    def hasDB(self, dbURL, dbEngine='sqlite3'):

        if dbEngine == 'sqlite3':
            try:
                open(dbURL, 'r')
                return True
            except FileNotFoundError:
                return False

        return False

    def getDB(self, dbURL):
        return SqliteDatabase(dbURL)

    def getEstimatorMeta(self, name):
        with db:
            print(f"{name} check 33 e dhukse ")
            res = EstimatorMeta.select().where(EstimatorMeta.name == name).get()
            print("res thik ase")
            return res

    def getTrainingProfile(self, name):
        with db:
            return TrainingProfile.select().where(TrainingProfile.name == name).get()

    def createEstimator(self, name, estimatorType, formula=None, loss=None, lr=0.001, optimizer=None, regularizer=None):
        with db:
            estimatorMeta = EstimatorMeta.create(name=name,
                                                 estimatorType=estimatorType,
                                                 formula=formula,
                                                 loss=loss,
                                                 lr=lr,
                                                 optimizer=optimizer,
                                                 regularizer=regularizer)
            print("in createMethod")
            if estimatorType == "LR":
                LRManager().create(name)
            elif estimatorType == "KNN":
                KNNManager().create(name)

            else:
                estimatorMeta.delete()
                raise Exception(f"unrecognized estimator type {estimatorType}")

            estimatorMeta.isAvailable = True
            estimatorMeta.trainable = True
            estimatorMeta.save()
            return estimatorMeta

    def createTrainingProfile(self, name, sql, validationSplit, batchSize, epoch, shuffle):

        with db:
            trainingProfile = TrainingProfile.create(name=name,
                                                     source=sql,
                                                     sourceType='sql',
                                                     validationSplit=validationSplit,
                                                     batchSize=batchSize,
                                                     epoch=epoch,
                                                     shuffle=shuffle
                                                     )

            return trainingProfile

    def train(self, currentDB, estimatorName, trainingProfileName):

        try:
            # 1 Check DB
            if currentDB is None:
                raise Exception(f"no Database chosen to draw training data from. hint: [USE DBUrl;]")

            # 2 Check Estimator
            estimatorMeta = self.getEstimatorMeta(estimatorName)
            if estimatorMeta.trainable == False:
                raise Exception(f"Estimator {estimatorMeta.name} is not trainable. Try cloning?")

            # 3 Get training profile
            trainingProfile = self.getTrainingProfile(trainingProfileName)

            # 4 Prepared data with formula processor and Train
            return self.prepareDataAndTrain(currentDB, estimatorMeta, trainingProfile)

        except EstimatorMeta.DoesNotExist as e:
            raise Exception(f"{estimatorName} estimator does not exist ({e}).")

        except TrainingProfile.DoesNotExist as e:
            raise Exception(f"{trainingProfileName} estimator does not exist ({e}.")

    def prepareDataAndTrain(self, currentDB, estimatorMeta, trainingProfile):
        # 1 Prepared data with formula processor
        formulaProcessor = FormulaProcessor(estimatorMeta.formula)
        XTrain, XValidation, yTrain, yValidation = formulaProcessor.getDataFromSQLDB(currentDB, trainingProfile)

        self.pp.pprint("Running training with following configurations.")
        self.pp.pprint(estimatorMeta)
        self.pp.pprint(trainingProfile)

        # 2 Train estimator with the data.
        if estimatorMeta.estimatorType == 'LR':
            estimatorManager = LRManager()
            accuracyDic = estimatorManager.trainValidate(estimatorMeta.name, XTrain, XValidation, yTrain, yValidation)
            self.postTrain(estimatorMeta)
            self.pp.pprint(accuracyDic)
            return accuracyDic
        elif estimatorMeta.estimatorType == 'KNN':
            estimatorManager = KNNManager()  # Assuming KNNManager is correctly implemented
            accuracyDic = estimatorManager.trainValidate(estimatorMeta.name, XTrain, XValidation, yTrain, yValidation)
            self.postTrain(estimatorMeta)
            self.pp.pprint(accuracyDic)
            return accuracyDic
        else:
            raise Exception(
                f"Unrecognized estimator type {estimatorMeta.estimatorType}. Only 'LR' and 'KNN' are supported.")

    def postTrain(self, estimatorMeta, stillTrainable=False):
        estimatorMeta.trainable = stillTrainable
        estimatorMeta.save()

    def cloneModel(self, fromName, toName, keepWeights=False):

        fromEstimatorMeta = self.getEstimatorMeta(fromName)

        if fromEstimatorMeta.estimatorType == 'LR':
            estimatorManager = LRManager()

            if estimatorManager.clonable == False:
                raise Exception(f"{fromEstimatorMeta.estimatorType} estimators cannot be cloned")
            else:
                estimatorMeta = self.createEstimator(name=toName,
                                                     estimatorType=fromEstimatorMeta.estimatorType,
                                                     formula=fromEstimatorMeta.formula,
                                                     loss=fromEstimatorMeta.loss,
                                                     lr=fromEstimatorMeta.lr,
                                                     optimizer=fromEstimatorMeta.optimizer,
                                                     regularizer=fromEstimatorMeta.regularizer
                                                     )
                return estimatorMeta
        pass

    def predict(self, currentDB, estimatorName, sql=None, trainingProfileName=None):
        try:
            # 1 Check DB
            if currentDB is None:
                raise Exception(f"no Database chosen to draw training data from. hint: [USE DBUrl;]")
            print(f"estimator name holo {estimatorName}")
            # 2 Check Estimator
            estimatorMeta = self.getEstimatorMeta(estimatorName)
            print("156 e passed")
            if estimatorMeta.isAvailable == False:
                raise Exception(f"Estimator {estimatorMeta.name} is not availble for use.")
            print("current db passed")
            if trainingProfileName is not None:
                print(" with training profile name")
                return self.predictWithTrainingProfile(currentDB, estimatorMeta, trainingProfileName)
            else:
                print("in with sql")
                return self.predictWithSQL(currentDB, estimatorMeta, sql)
        except EstimatorMeta.DoesNotExist as e:
            raise Exception(f"{estimatorName} estimator does not exist ({e}).")

        except TrainingProfile.DoesNotExist as e:
            raise Exception(f"{trainingProfileName} estimator does not exist ({e}.")

    def predictWithTrainingProfile(self, currentDB, estimatorMeta, trainingProfileName):
        trainingProfile = self.getTrainingProfile(trainingProfileName)
        if trainingProfile.sourceType == 'sql':
            print("in function of training")
            return self.predictWithSQL(currentDB, estimatorMeta, trainingProfile.source)
        else:
            raise NotImplementedError("prediction with non-sql training profile not implemented yet.")

    def plot_actual_vs_predicted(self, y_actual, y_predicted):
        plt.figure(figsize=(8, 6))
        plt.scatter(y_actual, y_predicted, color='blue')
        plt.plot(y_actual, y_actual, color='red')  # Plotting the ideal line where actual = predicted
        plt.title('Actual vs. Predicted')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return plot_data

    def predictWithSQL(self, currentDB, estimatorMeta, sql):
        # Assume df and X are obtained from FormulaProcessor as before
        df, X = FormulaProcessor(estimatorMeta.formula).getDfAndXFromSQL(currentDB, sql, onlyPredictors=True)
        y_actual = FormulaProcessor(estimatorMeta.formula).getYfromSQL(currentDB, sql)
        if estimatorMeta.estimatorType == 'LR':
            estimatorManager = LRManager()
            if not estimatorManager.isFitted(estimatorMeta.name):  # You need to implement this method
                raise Exception(f"Model {estimatorMeta.name} is not fitted. Please train the model before prediction.")
            predictions = estimatorManager.predict(estimatorMeta.name, X)

            df['prediction'] = predictions
            df['actual']=y_actual

            df = pd.DataFrame(df)
            df = df.to_dict(orient='records')

            plot_data = self.plot_actual_vs_predicted(y_actual,predictions)
            df_summary = pd.DataFrame({"Actual Values": y_actual, "Predicted Values": predictions}).to_dict(orient='records')
            result={'table':df,
                    'graph':plot_data,
                    'text':'',
                    'actvspred':df_summary
                    }
            return result
        elif estimatorMeta.estimatorType == 'KNN':
            estimatorManager = KNNManager()
            if not estimatorManager.isFitted(estimatorMeta.name):
                raise Exception(f"Model {estimatorMeta.name} is not fitted. Please train the model before prediction.")
            predictions = estimatorManager.predict(estimatorMeta.name, X)
            df['prediction'] = predictions
            df['actual'] = y_actual
            df = pd.DataFrame(df)
            df = df.to_dict(orient='records')
            plot_data = self.plot_actual_vs_predicted(y_actual, predictions)
            df_summary = pd.DataFrame({"Actual Values": y_actual, "Predicted Values": predictions}).to_dict(orient='records')
            result = {'table': df,
                      'graph': plot_data,
                      'text': '',
                      'actvspred': df_summary
                      }
            return result
        else:
            raise ("Model is not fitted. or did't found the model")

    @staticmethod
    def create_table(db, table_name, columns):
        class Meta:
            database = db

        attrs = {'__module__': __name__, 'Meta': Meta}
        for column_name, data_type in columns:
            if data_type.lower() == 'int':
                attrs[column_name] = IntegerField()
            elif data_type.lower() == 'float':
                attrs[column_name] = FloatField()
            else:
                attrs[column_name] = TextField()

        table_class = type(table_name, (Model,), attrs)
        db.create_tables([table_class])
        print(f"Table {table_name} created successfully.")