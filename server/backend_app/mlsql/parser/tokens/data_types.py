dataTypeTokens = [
    'WORD',
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