from peewee import *
import datetime
from .db import db

class TrainingProfile(Model):
    name = CharField(primary_key=True)
    dateCreated = DateTimeField(default=datetime.datetime.now)
    source = CharField()
    sourceType = CharField(default='sql')
    validationSplit = FloatField(default=0)
    batchSize = IntegerField(default=0)
    epoch = IntegerField(default=1)
    shuffle = BooleanField(default=False)

    class Meta:
        database = db
    
    def __str__(self):
        return f"""name: {self.name}, 
dateCreated: {self.dateCreated},
source: {self.source},
sourceType: {self.sourceType},
validationSplit: {self.validationSplit},
batchSize: {self.batchSize},
epoch: {self.epoch},
shuffle: {self.shuffle}
"""
