# DL4ML
Machine Learning Using Declarative Language
# Dependency requirements: (will be packaged together in future)
1. Tensorflow 1.2+
2. Python 3.6
3. Python Ply package
4. sci-kit learn package
5. Pandas package
6. Dill package


## Run the application
for frontend:
```
    npm install
    npm  run dev
```
for backend: 
```
    pip install -r requirements.txt
    python manage.py runserver
```


## Creating an estimator
Syntax: CREATE ESTIMATOR **estimator_name** TYPE **type_name** FORMULA $**formula**$;

Example: CREATE ESTIMATOR **salaryPredictor** TYPE **LR FORMULA** $**salary~years**$;
### Formula

Formula fields must exactly be the same as the SQL resultset.

## Creating a training profile

Syntax: CREATE TRAINING PROFILE **profile_name** WITH [**sql**];

Example: CREATE TRAINING PROFILE **oneshotSalary** WITH [**SELECT * FROM salary**];

## Choose Database
USE '**data/salarydb.db**';

## Training an estimator with a training profile
Syntax: TRAIN **estimator_name** WITH TRAINING PROFILE **profile_name**;

Example: TRAIN **salaryPredictor** WITH TRAINING PROFILE **oneshotSalary**;

Syntax: TRAIN **estimator_name** WITH **profile_name**;

Example: TRAIN **salaryPredictor** WITH **oneshotSalary**;

## Predicting

Pre-requisite:
- First choose the source database 
- Make sure the model is trained

1. Predicting with existing training profile

Syntax: PREDICT WITH TRAINING PROFILE **profile_name** BY ESTIMATOR **estimator_name**;

Example: PREDICT WITH TRAINING PROFILE **oneshotSalary** BY ESTIMATOR **salaryPredictor**;

2. Predicting with and SQL

Syntax: PREDICT WITH [**SQL**] BY ESTIMATOR **estimator_name**;

Example: PREDICT WITH [**SELECT * FROM salary**] BY ESTIMATOR **salaryPredictor**;


# Sample input as json format 
```
CREATE ESTIMATOR salaryPredictor TYPE LR FORMULA $salary~years$;
CREATE TRAINING PROFILE oneshotSalary WITH [SELECT * FROM salary];
USE 'data/salarydb.db';"
TRAIN salaryPredictor WITH TRAINING PROFILE oneshotSalary;
PREDICT WITH TRAINING PROFILE oneshotSalary BY ESTIMATOR salaryPredictor;
```



# Advanced Usage:
1. Reusing a training profile

You may use the same training profile to train other models. Just use the training syntax.

2. Cloning an estimator

Cloning an estimator helps if you want to train a model with different datasets.

Syntax: CLONE ESTIMATOR **estimator_name** AS **new_estimator_name**;

Example: CLONE ESTIMATOR **salaryPredictor** AS **clonedSalaryPredictor**;


# Available ML tools:
1. Linear Regresstion (LR)



