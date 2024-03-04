import json
import os

from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from django.conf import settings

from .mlsql.data.setup import setup
from .mlsql.parser.parser import _parser
from .mlsql.test.csvToDB import csvToDB


# Create your views here.
@csrf_exempt
# Create your views here.
@api_view(['GET','POST'])
def initial(request):
    setup()
@api_view(['GET','POST'])
def parser_view(req):
    data=json.loads(req.body)
    data = data.get("inpt")
    return JsonResponse(_parser(data))
@api_view(['GET','POST'])
def test_view(req):
    setup()

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

    data = req.POST.get('input') #json.loads(req.body)
    last_response = None
    # Split the string by comma separator
    data = data.split(',')
    # Iterate over data
    for cmd in data:
        response_generator = _parser(cmd)
        # Iterate over response_generator and update last_response
        for response in response_generator:
            last_response = response #json.dumps(response) + "\n"

    # Check if last_response is not None
    if last_response is not None:
        return JsonResponse(last_response, safe=False)
    else:
        # Return appropriate response if no response is generated
        return JsonResponse({"message": "No predicted response generated"}, status=400)

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
CREATE ESTIMATOR salaryPredictor TYPE LR FORMULA $salary~years$;
CREATE TRAINING PROFILE oneshotSalary WITH [SELECT * FROM salary];"
USE 'data/salarydb.db';"
TRAIN salaryPredictor WITH TRAINING PROFILE oneshotSalary;"
PREDICT WITH TRAINING PROFILE oneshotSalary BY ESTIMATOR salaryPredictor;
"""
#sample input
"""
{
  "inpt": [
    "CREATE ESTIMATOR salaryPredictor TYPE LR FORMULA $salary~years$;",
    "CREATE TRAINING PROFILE oneshotSalary WITH [SELECT * FROM salary];",
    "USE 'data/salarydb.db';",
    " TRAIN salaryPredictor WITH TRAINING PROFILE oneshotSalary;",
    "PREDICT WITH TRAINING PROFILE oneshotSalary BY ESTIMATOR salaryPredictor;"
  ]
}
"""