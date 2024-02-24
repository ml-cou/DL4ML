projectFolder = 'D:/pyhton_project/Project/dl4ml/server/backend_app/mlsql'
import logging, sys, math,os
sys.path.append(str(projectFolder))


from .db import db
from .EstimatorMeta import EstimatorMeta
from .TrainingProfile import TrainingProfile

def setup():
    tables = [EstimatorMeta, TrainingProfile]
    db.connect()
    db.drop_tables(tables)
    db.create_tables(tables)
    db.close()
    print("setup complete")
    pass