import json

from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view

from .mlsql.data.setup import setup
from .mlsql.parser.parser import _parser


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
    data = json.loads(req.body)
    data=data.get("input")

    # for single command
    # res=list(_parser(data))
    # return JsonResponse(res,safe=False)

    #for multiple command
    def generate_responses():
        for cmd in data:
            response_generator = _parser(cmd)
            for response in response_generator:
                yield json.dumps(response) + "\n"

    return StreamingHttpResponse(generate_responses(), content_type="application/json")

# CREATE ESTIMATOR salaryPredictor TYPE LR FORMULA $salary~years$;
# CREATE TRAINING PROFILE oneshotSalary WITH [SELECT * FROM salary];
# USE 'data/salarydb.db';
# TRAIN salaryPredictor WITH TRAINING PROFILE oneshotSalary;
# PREDICT WITH TRAINING PROFILE oneshotSalary BY ESTIMATOR salaryPredictor;

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