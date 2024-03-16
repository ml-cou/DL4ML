import os

import ply.lex as lex
from ply.lex import TOKEN
import dill
from .tokens.manipulations import *
from .tokens.sql import *
from .tokens.definitions import *
from .tokens.data_types import *

#modified
# Add to the list of keywords if not already present
modelTokens += ['TABLE']
dataTypeTokens = [
    'WORD',
    # 'CHAR'
    'FLOAT',
    'INT',
    'DELIMITER',
    'BOOL',
    'URL'
]

#regular expressions
# t_BOOL = r'true|false'
t_WORD = r'[a-zA-Z_][a-zA-Z_0-9]*'
t_FLOAT = r'[0-9]+\.[0-9]*|[0-9]*\.[0-9]+'
t_INT = r'[0-9]+'
t_DELIMITER = r';'
# t_CHAR = r'[a-zA-Z_][a-zA-Z_0-9]*'
trainTokens = [
    'TRAIN',
    'WITH',
    'TRAINING_PROFILE'
]

utilityTokens = [
    'SET',
    'DEBUG',
    'AS'
]
modelTokens = [
    'CREATE',
    'CLONE',
    'MODEL',
    'ESTIMATOR',
    'REGULARIZER',
    'MODEL_TYPE',
    'FORMULA',
    'FORMULA_EXP',
    'LOSS',
    'LEARNING_RATE',
    'OPTIMIZER',
    'WEIGHTS',
]

trainProfileTokens = [
    'CREATE',
    'AND',
    'TRAINING_PROFILE',
    'BATCH_SIZE',
    'EPOCH',
    'SHUFFLE',
    'VALIDATION_SPLIT',
    'WITH',
    'USE',
    'BY'
]

trainTokens = [
    'TRAIN'
]

predictTokens = [
    'PREDICT',
    'TEST'
]

sqlKeywords = [
    'TABLE',
]

#-------modified end ----------
keywords = list(set().union(
            modelTokens,
            trainTokens,
            trainProfileTokens,
            predictTokens,
            utilityTokens,
            sqlKeywords
            ))
tokens =  list(set().union(
            dataTypeTokens,
            basicSQL
            )) + keywords
tokens += ['LPAREN', 'RPAREN']
t_LPAREN = r'\('
t_RPAREN = r'\)'
# print(tokens)



def t_BOOL(t):
    r'true|false'
    if t.value == 'true':
        t.value = True
    else:
        t.value = False
    
    return t


def t_SQL(t):
    r'\[\s*[SELECT,UPDATE]+[^\];]*\]'
    t.value = t.value[1:len(t.value)-1].strip()
    return t

def t_URL(t):
    r'\'[a-z0-9A-Z\.\/\:\%\+\-\_\&\@ ]+\''
    t.value = t.value[1:len(t.value)-1].strip()
    return t

def t_FORMULA_EXP(t):
    r'\$[a-zA-Z_0-9]+\~[a-zA-Z_0-9\+\-]+\$'
    t.value = t.value[1:len(t.value)-1].strip()
    return t
    
# keywords rule
# reKyewords = "(" + "|".join(keywords) + ")+[ \n\t]{1}"
reKyewords = r'\b(' + '|'.join(keywords) + r')\b'

@TOKEN(reKyewords)
def t_KEYWORD(t):
    # print(f"found keyword: {t.value}")
    t.value = t.value.strip()
    t.type = t.value
    return t
 
# Define a rule so we can track line numbers
def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

def t_TRAINING_PROFILE(t):
    r'TRAINING[_ \t\n]+PROFILE'
    t.type = 'TRAINING_PROFILE'
    t.value = 'TRAINING_PROFILE'
    # print("found t_TRAINING_PROFILE")
    return t


def t_LEARNING_RATE(t):
    r'LEARNING[_ \t\n]+RATE'
    t.type = 'LEARNING_RATE'
    t.value = 'LEARNING_RATE'
    # print("found t_LEARNING_RATE")
    return t

def t_VALIDATION_SPLIT(t):
    r'VALIDATION[_ \t\n]+SPLIT'
    t.type = 'VALIDATION_SPLIT'
    t.value = 'VALIDATION_SPLIT'
    # print("found t_VALIDATION_SPLIT")
    return t

def t_FORMULA_OPERATOR(t):
    r'~'
    t.type = 'FORMULA_OPERATOR'
    t.value = 'FORMULA_OPERATOR'
    # print("found t_FORMULA_OPERATOR")
    return t

    
# A string containing ignored characters (spaces and tabs)
t_ignore  = ' \t'

# Error handling rule
def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)

lexer = lex.lex()
current_directory = os.path.dirname(__file__)
currentDBURL = os.path.join(current_directory,"lexer.dill")
with open(currentDBURL, "wb") as f:
    dill.dump(lexer, f)