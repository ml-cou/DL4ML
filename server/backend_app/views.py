import json
import os

from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from django.conf import settings

from .mlsql.data.setup import setup
from .mlsql.parser.parser import _parser
from .mlsql.parser.query_process import query_process
from .mlsql.test.csvToDB import csvToDB


# Create your views here.
@csrf_exempt
# Create your views here.
@api_view(['GET','POST'])
def initial(request):
    setup()
from django.http import JsonResponse
import json
@api_view(['GET', 'POST'])
def test_view(req):
    if req.method == 'POST' and req.FILES['file']:
        file = req.FILES['file']
        file_name = file.name
        if file_name.endswith('.csv'):
            csvToDB(file)
        else:
            current_directory = os.path.dirname(__file__)
            file_path = os.path.join(current_directory, f'./mlsql/data/{file_name}')
            # Open the file and write the uploaded content to it
            with open(file_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)

    data = req.POST.get('input')  # json.loads(req.body)
    data = data.strip()
    data = data.split(';')
    print(data)
    responses = {}  # Store responses in a dictionary
    for index, cmd in enumerate(data):
        if cmd != '':
            response_key = f'response_{index}'  # Generate a unique key for each response
            responses[response_key] = {}  # Initialize list for each response
            # Iterate over the generator object returned by query_process
            for response_dict in query_process(cmd):
                responses[response_key].update(response_dict)  # Append yielded dictionaries to the list

    print(responses)
    # Now responses is a dictionary of lists (each list contains dictionaries), which can be serialized to JSON
    response_json = json.dumps(responses)

    print("response is", response_json)
    return JsonResponse(response_json, safe=False)

@api_view(['GET','POST'])
def parser_view(req):
    data=json.loads(req.body)
    data = data.get("inpt")
    return JsonResponse(_parser(data))

@api_view(['GET','POST'])
def upload(req):
    if req.method == 'POST' and req.FILES['file']:
        file = req.FILES['file']
        file_name = file.name
        current_directory = os.path.dirname(__file__)
        file_path = os.path.join(current_directory, file_name)
        # Open the file and write the uploaded content to it
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
        return HttpResponse('File uploaded successfully!')

    return HttpResponse('No file uploaded.')


"""
CREATE ESTIMATOR salaryPredictor MODEL_TYPE LR FORMULA $salary~years$;
CREATE TRAINING PROFILE oneshotSalary WITH [SELECT * FROM salary];
USE 'data/salarydb.db';
TRAIN salaryPredictor WITH TRAINING PROFILE oneshotSalary;
PREDICT WITH TRAINING PROFILE oneshotSalary BY ESTIMATOR salaryPredictor;

=====================TASK===========================

CREATE MODEL model_name MODEL_TYPE LR FORMULA tax FOR $feature 1, $feature 2,...... 
USE DB 
PREDICT BY MODEL_NAME  WITH TABLE_NAME 


create model taxpred as formula tax, state, size for age, profession, school [visualize line];
predict formula WITH model_name USING table_name ;
VISUALIZE 

=======================================
CREATE TABLE Persons (PersonIDint ,LastName varchar(255),FirstName varchar(255),Address varchar(255),City varchar(255));
CREATE ESTIMATOR salaryPredictor MODEL_TYPE LR FORMULA $tax~rad+dis+ptratio$;
CREATE TRAINING PROFILE oneshotSalary WITH [SELECT * FROM Boston];
USE 'data/Boston.db';
TRAIN salaryPredictor WITH TRAINING PROFILE oneshotSalary;
PREDICT WITH TRAINING PROFILE oneshotSalary BY ESTIMATOR salaryPredictor;

CREATE TABLE customer '(' id,name  ')'

CREATE ESTIMATOR salaryPredictor MODEL_TYPE KNN FORMULA $Species~SepalLengthCm$;
CREATE TRAINING PROFILE oneshotSalary WITH [SELECT * FROM Iris];
USE 'data/Iris.db';
TRAIN salaryPredictor WITH TRAINING PROFILE oneshotSalary;
PREDICT WITH TRAINING PROFILE oneshotSalary BY ESTIMATOR salaryPredictor;
"""
#sample input
"""
{
  "inpt": [
    "CREATE ESTIMATOR salaryPredictor MODEL_TYPE LR FORMULA $salary~years$;",
    "CREATE TRAINING PROFILE oneshotSalary WITH [SELECT * FROM salary];",
    "USE 'data/salarydb.db';",
    " TRAIN salaryPredictor WITH TRAINING PROFILE oneshotSalary;",
    "PREDICT WITH TRAINING PROFILE oneshotSalary BY ESTIMATOR salaryPredictor;"
  ]
}


    Creates an estimator using the specified parameters.

    Args:
        name (str): Name of the estimator.
        estimatorType (str): Type of estimator (e.g., 'linear_regression').
        formula (str): Formula specifying the target variable and features in the format:
                       "target_variable ~ feature1 + feature2 + ... + featureN"
        loss (str, optional): Loss function (e.g., 'mean_squared_error'). Defaults to None.
        lr (float, optional): Learning rate (default: 0.001).
        optimizer (str, optional): Optimizer (e.g., 'adam'). Defaults to None.
        regularizer (str, optional): Regularizer (e.g., 'l2'). Defaults to None.

    Returns:
        EstimatorMeta: The created estimator meta object.

"""

# @api_view(['GET','POST'])
# def test_view(req):
#     # setup()
#
#     if req.method == 'POST' and req.FILES['file']:
#         file = req.FILES['file']
#         file_name = file.name
#         if file_name.endswith('.csv'):
#             csvToDB(file)
#         else:
#             current_directory = os.path.dirname(__file__)
#             file_path = os.path.join(current_directory, f'./mlsql/data/{file_name}')
#             # Open the file and write the uploaded content to it
#             with open(file_path, 'wb+') as destination:
#                 for chunk in file.chunks():
#                     destination.write(chunk)
#
#     data = req.POST.get('input') #json.loads(req.body)
#     last_response = None
#     print(data)
#     # Split the string by comma separator
#     data = data.split(';')
#     # Iterate over data
#     for cmd in data:
#         response_generator = _parser(cmd)
#         # Iterate over response_generator and update last_response
#         for response in response_generator:
#             last_response = response #json.dumps(response) + "\n"
#         print("cmd is ===========",cmd)
#     print("last respons=========e",response_generator)
#     print("response end ===========")
#     # Check if last_response is not None
#     if last_response is not None:
#         return JsonResponse(last_response, safe=False)
#     else:
#         # Return appropriate response if no response is generated
#         return JsonResponse({"message": "No predicted response generated"}, status=400)
