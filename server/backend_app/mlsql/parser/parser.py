
import logging, sys, math,os
import traceback

import pandas as pd
# current_directory = os.path.dirname(__file__)
# currentDBURL = os.path.join(current_directory,"lexer.dill")
# print("Current directory is: " + currentDBURL)
# projectFolder = 'D:/pyhton_project/Project/dl4ml/server/backend_app/mlsql'
# # currentFolder = os.path.abspath('')
# # try:
# #     sys.path.remove(str(currentFolder))
# # except ValueError: # Already removed
# #     pass
#
# sys.path.append(str(projectFolder))

# logging.basicConfig(
#     level = logging.DEBUG,
#     filename = "parselog.txt",
#     filemode = "w",
#     format = "%(filename)10s:%(lineno)4d:%(message)s"
# )
# log = logging.getLogger()

import ply.yacc as yacc
import dill
from .lexer import tokens
from ..engine.ASTProcessor import ASTProcessor
from ..data.setup import setup
#******Environment Setup******

ASTProcessor = ASTProcessor()
ESTIMATOR, TRAIN, TRAINING_PROFILE, USE, PREDICT = range(5)
states = ['ESTIMATOR', 'TRAIN', 'TRAINING_PROFILE', 'USE', 'PREDICT' ]
currentState = None
currentDB = None #database connector instance, not url.

DEBUG = True
def setDebug(debug = False):
    global DEBUG
    DEBUG = debug

#******Environment Setup Ends******
global response
response=''
#***********Grammar******************
def p_create_model(p):
    '''exp : CREATE ESTIMATOR WORD MODEL_TYPE WORD FORMULA FORMULA_EXP DELIMITER
            | CREATE ESTIMATOR WORD MODEL_TYPE WORD FORMULA FORMULA_EXP LOSS WORD DELIMITER
            | CREATE ESTIMATOR WORD MODEL_TYPE WORD FORMULA FORMULA_EXP LOSS WORD LEARNING_RATE FLOAT DELIMITER
            | CREATE ESTIMATOR WORD MODEL_TYPE WORD FORMULA FORMULA_EXP LOSS WORD LEARNING_RATE FLOAT OPTIMIZER WORD REGULARIZER WORD DELIMITER'''
    setup()
    printMatchedRule('p_create_model')
    global currentState, formula
    currentState = ESTIMATOR

    length = len(p)
    for i in range(length):
        print("p is =",p[i])
    name = p[3]
    estimatorType = p[5]
    loss = None
    lr = 0.001
    optimizer = None
    regularizer = None

    lastPos = 5
    if length > 7:
        lastPos += 2
        formula = p[lastPos]

    if length > lastPos + 2:
        lastPos += 2
        loss = p[lastPos]

    if length > lastPos + 2:
        lastPos += 2
        lr = p[lastPos]

    if length > lastPos + 2:
        lastPos += 2
        optimizer = p[lastPos]
        lastPos += 2
        regularizer = p[lastPos]
    print("in create function \n ",name,estimatorType,formula,loss,lr,optimizer,regularizer)
    try:
        estimator = ASTProcessor.createEstimator(name=name, 
                                                estimatorType=estimatorType, 
                                                formula=formula, 
                                                loss=loss, 
                                                lr=lr, 
                                                optimizer=optimizer, 
                                                regularizer=regularizer)
        print(f"Created estimator:\n{estimator}")
        global response
        response=f"Created estimator:\n{estimator}"
    except Exception as e:
        printError(e)
    pass

def p_training_profile(p):
    '''exp : CREATE TRAINING_PROFILE WORD WITH SQL DELIMITER
                | CREATE TRAINING_PROFILE WORD WITH SQL AND VALIDATION_SPLIT FLOAT DELIMITER
                | CREATE TRAINING_PROFILE WORD WITH SQL AND VALIDATION_SPLIT FLOAT BATCH_SIZE INT EPOCH INT DELIMITER
                | CREATE TRAINING_PROFILE WORD WITH SQL AND VALIDATION_SPLIT FLOAT BATCH_SIZE INT EPOCH INT SHUFFLE BOOL DELIMITER'''
    printMatchedRule('p_training_profile')
    global currentState
    currentState = TRAINING_PROFILE
    length = len(p)

    name = p[3]
    sql = p[5]
    validationSplit = 0
    batchSize = 0
    epoch = 0
    shuffle = False

    if length > 7:
        validationSplit = p[8]
    if length > 10:
        batchSize = p[10]
        epoch = p[12]
    if length > 14:
        shuffle = p[14]
    print("in train function \n ", name, "\n ", sql, "\n ",)
    try:
        profile = ASTProcessor.createTrainingProfile(name=name, sql=sql, validationSplit=validationSplit, batchSize=batchSize, epoch=epoch, shuffle=shuffle)
        print(f"Created training profile:\n{profile}")
        global  response
        response=f"Created training profile:\n{profile}"
    except Exception as e:
        printError(e)
    pass

def p_train(p):
    '''exp : TRAIN WORD WITH WORD DELIMITER
           | TRAIN WORD WITH TRAINING_PROFILE WORD DELIMITER'''
    printMatchedRule('p_train')
    global currentState, currentDB
    currentState = TRAIN

    length = len(p)
    estimatorName = p[2]
    
    if length == 6:
        trainingProfileName = p[4]
    else:
        trainingProfileName = p[5]

    try:
        ASTProcessor.train(currentDB, estimatorName, trainingProfileName)
    except Exception as e:
        printError(e)
    pass

def p_predict(p):
    '''exp : PREDICT WITH SQL BY ESTIMATOR WORD DELIMITER
           | PREDICT WITH TRAINING_PROFILE WORD BY ESTIMATOR WORD DELIMITER'''
    printMatchedRule('p_predict')
    global currentState, currentDB
    currentState = PREDICT

    length = len(p)

    sql = p[3]
    estimatorName = p[6]
    trainingProfileName = None
    if p[3] == 'TRAINING_PROFILE':
        sql = None
        estimatorName = p[7]
        trainingProfileName = p[4]

    try:
        global  response
        response = ASTProcessor.predict(currentDB, estimatorName,sql=sql, trainingProfileName=trainingProfileName )

        # print(df.to_string()) # TODO, use an internal state in parser and print with arrows and exit commands.
    except Exception as e:
        printError(e)
    pass

def p_clone_model(p):
    '''exp : CLONE ESTIMATOR WORD AS WORD DELIMITER
           | CLONE ESTIMATOR WORD AS WORD WITH WEIGHTS DELIMITER'''
    printMatchedRule('p_clone_model')
    global currentState
    currentState = ESTIMATOR

    length = len(p)
    fromName = p[3]
    toName = p[5]
    keepWeights = False

    if length > 7:
        if p[7] == 'WEIGHTS':
            keepWeights = True

    try:
        estimatorMeta = ASTProcessor.cloneModel(fromName, toName, keepWeights)
        print(f"Cloned estimator:{estimatorMeta}")
    except Exception as e:
        printError(e)
    pass

def p_use_database(p):
    'exp : USE URL DELIMITER'
    printMatchedRule('p_use_database')
    global currentDB, response
    current_directory = os.path.dirname(__file__)
    currentDBURL = os.path.join(current_directory,f'../{p[2]}')

    if ASTProcessor.hasDB(currentDBURL):
        print(f"selected {currentDBURL}")
        response = {'text': f"selected {p[2]}"}
        currentDB = ASTProcessor.getDB(currentDBURL)
    else:
        response = {'text': f'\'{p[2]}\' does not exist in the database engine.'}
        printError(f'\'{currentDBURL}\' does not exist in the database engine.')
def p_SQL(p):
    'exp : SQL DELIMITER'
    printMatchedRule('p_SQL')
    p[0] = p[1]
    print( f" p[0] = {p[0]}" )

    pass

def p_DEBUG(p):
    'exp : SET DEBUG BOOL DELIMITER'
    printMatchedRule('p_DEBUG')
    global DEBUG
    DEBUG = p[3]
    # if p[3] == 'true':
    #     DEBUG = True
    # else:
    #     DEBUG = False
    
    print(f"Debug set to {p[3]}")

    pass

def p_error(t):
    printError('Syntax error at "%s"' % t.value if t else 'NULL')
    global current_state
    current_state = None
    pass

def printError(e):
    global DEBUG
    print("Error:", end=" ")
    print(e)

    if DEBUG:
        print("strack trace:")
        for line in traceback.format_stack():
            print(line.strip())

def printMatchedRule(rule):
    if DEBUG:
        print(f"Matched Grammar Rule: {rule}")

#***********Grammar Ends******************
# parser = yacc.yacc(debug=True, errorlog=log)
parser = yacc.yacc(debug=True)

# with open("parser/parser.dill", "wb") as f:
#     dill.dump(parser, f)

def welcome():
    print(f'''
******** Welcome to MLSQL (version egg)*******
The first open-source SQL for Machine Learning
**********************************************
''')
    pass

def p_create_table(p):
    '''exp : CREATE TABLE WORD DELIMITER'''
    printMatchedRule('p_create_table')
    print("in table")
    # pass
    table_name = p[3]
    columns = p[5]  # This will be a list of tuples [(column_name, data_type), ...]

    try:
        ASTProcessor.create_table(currentDB, table_name, columns)
        print(f"Table {table_name} created with columns {columns}")
    except Exception as e:
        printError(e)

def p_table_columns(p):
    '''table_columns : table_column ',' table_column
                     | table_column'''
    if len(p) == 4:
        p[0] = p[1] + [p[3]]
    else:
        p[0] = [p[1]]

def p_table_column(p):
    '''table_column : WORD data_type'''
    p[0] = (p[1], p[2])


def p_data_type(p):
    '''data_type : INT
                 | FLOAT
                 | CHAR
                 | TEXT'''  # Add more data types as needed
    p[0] = p[1]



def _parser(cmd):
    # welcome()
    # print("Current directory is: " + currentDBURL)

    userInput  = ''
    prevInput = ''
    # print("\033[32m",'MLSQL>', end=" ")
    print('MLSQL>', end=" ")
    # for cmd in sql_commands:
    userInput = cmd  #input().strip()

    # if userInput == 'exit':
    #     break
    #
    # if len(userInput) == 0:
    #     continue
    #
    #
    # if userInput[-1] != ';':
    #     prevInput += ' ' + userInput
    #     continue
    data = prevInput + userInput
    print(f"parsing {data}")
    p = parser.parse(data)
    print(p)
    prevInput = ''
    # for single command
    # global response
    return response
    #for multiple command
    # yield response
        # pass

