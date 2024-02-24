from peewee import *
import datetime
from .db import db

class EstimatorMeta(Model):
    name = CharField(primary_key=True)
    dateCreated = DateTimeField(default=datetime.datetime.now)
    isAvailable = BooleanField(default=False)
    estimatorType = CharField()
    formula = CharField(null=True)
    loss = CharField(null=True)
    lr = FloatField(null=True)
    optimizer = CharField(null=True)
    regularizer = CharField(null=True)
    trainable = BooleanField(default=True)

    class Meta:
        database = db
    
    def __str__(self):
        return f"""name: {self.name}, 
estimatorType: {self.estimatorType}, 
dateCreated: {self.dateCreated}, 
isAvailable: {self.isAvailable}, 
formula: {self.formula}, 
loss: {self.loss}, 
lr: {self.lr}, 
optimizer: {self.optimizer}, 
regularizer: {self.regularizer}, 
trainable: {self.trainable}
"""