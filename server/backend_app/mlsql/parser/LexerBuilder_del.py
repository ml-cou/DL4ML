import ply.lex as lex
import dill
from tokens.definitions import *
from tokens.data_types import *
from tokens.manipulations import *
from tokens.sql import *

class LexerBuilder:

    tokens = ()

    def __init__(self):

        LexerBuilder.tokens = list(set().union(

            modelTokens,
            dataTypeTokens,
            trainTokens,
            trainProfileTokens,
            basicSQL
            ))

        pass
    def build( self, **kwargs):

        print( LexerBuilder.tokens )
        self.lexer = lex.lex(module=self, **kwargs)
        with open("lexer.dill", "wb") as f:
            dill.dump(self.lexer, f)
        return self.lexer

    

    #regular expressions

    t_ALPHA_NUMERIC = r'[a-zA-Z_][a-zA-Z_0-9]*'
    def t_CREATE(self, t):
        r'CREATE'
        return t
    def t_MODEL(self, t):
        r'MODEL'
        return t
    def t_REGULARIZER(self, t):
        r'REGULARIZER'
        return t
    def t_TYPE(self, t):
        r'TYPE'
        return t
        
    
    # Define a rule so we can track line numbers
    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)

    # A string containing ignored characters (spaces and tabs)
    t_ignore  = ' \t'

    # Error handling rule
    def t_error(self, t):
        print("Illegal character '%s'" % t.value[0])
        t.lexer.skip(1)