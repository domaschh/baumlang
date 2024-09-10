import re
from enum import Enum, auto

class TokenType(Enum):
    LET = auto()
    IDENTIFIER = auto()
    NUMBER = auto()
    STRING = auto()
    ASSIGN = auto()
    EQUALS = auto()
    NOT = auto()
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    IF = auto()
    ELSE = auto()
    GREATER = auto()
    LESS = auto()
    SEMICOLON = auto()
    PRINT = auto()
    PIPE = auto()

class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __repr__(self):
        return f"({self.type}, {self.value})"

    def is_operator(self):
        return self.type == TokenType.PLUS or self.type == TokenType.MINUS or self.type == TokenType.MULTIPLY or self.type == TokenType.DIVIDE

def tokenize(code):
    tokens = []
    keywords = {
        'let': TokenType.LET,
        'if': TokenType.IF,
        'else': TokenType.ELSE
    }

    token_specifications = [
        ('NUMBER', r'\d+(\.\d*)?'),
        ('STRING', r'"[^"]*"'),
        ('IDENTIFIER', r'[a-zA-Z_]\w*'),
        ('ASSIGN', r'='),
        ('EQUALS', r'=='),
        ('NOT', r'!'),
        ('PLUS', r'\+'),
        ('MINUS', r'-'),
        ('MULTIPLY', r'\*'),
        ('DIVIDE', r'/'),
        ('LPAREN', r'\('),
        ('RPAREN', r'\)'),
        ('LBRACE', r'\{'),
        ('RBRACE', r'\}'),
        ('LBRACKET', r'\['),
        ('RBRACKET', r'\]'),
        ('COMMA', r','),
        ('GREATER', r'>'),
        ('LESS', r'<'),
        ('SEMICOLON', r';'),
        ('PIPE', r'\|'),
        ('WHITESPACE', r'\s+'),
    ]

    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specifications)
    line_num = 1
    line_start = 0
    for mo in re.finditer(tok_regex, code):
        kind = mo.lastgroup
        value = mo.group()
        column = mo.start() - line_start
        if kind == 'NUMBER':
            value = float(value) if '.' in value else int(value)
            tokens.append(Token(TokenType.NUMBER, value))
        elif kind == 'STRING':
            tokens.append(Token(TokenType.STRING, value[1:-1]))
        elif kind == 'IDENTIFIER':
            if value in keywords:
                tokens.append(Token(keywords[value], value))
            else:
                tokens.append(Token(TokenType.IDENTIFIER, value))
        elif kind == 'WHITESPACE':
            if '\n' in value:
                line_start = mo.end()
                line_num += value.count('\n')
        else:
            tokens.append(Token(TokenType[kind], value))

    return tokens



class Statement:
    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(f'{k}={v!r}' for k, v in self.__dict__.items())})"

class Eval(Statement):
    pass

class Decl(Statement):
    def __init__(self, name):
        self.name = name

class FuncDecl(Decl):
    def __init__(self, name, args, body):
        super().__init__(name)
        self.args = args
        self.body = body

class VarDecl(Decl):
    def __init__(self, name, expr):
        super().__init__(name)
        self.expr = expr

class Expr(Statement):
    pass

class Program:
    def __init__(self, stmts):
        self.statements = stmts

    def execute(self):
        for stmt in self.statements:
            if isinstance(stmt, Eval):
                self.eval(stmt)
            else:
                continue

    def eval(self, stmt):
        print("evaluating")

class ListExpr(Expr):
    def __init__(self, elements):
        self.elements = elements

class BinaryExpr(Expr):
    def __init__(self, lhs, rhs, operator):
        self.rhs = rhs
        self.lhs = lhs
        self.operator = operator


class NumberLiteral(Expr):
    def __init__(self, value):
        self.value = value

class Variable(Expr):
    def __init__(self, name):
        self.name = name

class Parser:
    def __init__(self, tkns):
        self.tokens = tkns
        self.idx = 0

    def tokens_left(self):
        if self.idx < len(self.tokens):
            return True
        else:
            return False

    def peek(self, n) -> Token:
        return self.tokens[self.idx + n]

    def consume(self) -> Token:
        self.idx = self.idx + 1
        return self.tokens[self.idx - 1]

    def current_token(self) -> Token:
        return self.tokens[self.idx]

    def parse(self) -> [Statement]:
        statements = []
        while self.tokens_left():
            statements.append(self.parse_statement())
        return statements

    def parse_expr(self, precedence=0) -> Expr:
        if self.current_token().type == TokenType.LBRACKET:
            return self.parse_list_expr()

        left = self.parse_primary()
        while self.tokens_left() and self.current_token().is_operator() and self.get_precedence(self.current_token()) > precedence:
            left = self.parse_binary_expr(left, self.get_precedence(self.current_token()))

        return left

    def parse_primary(self):
        token = self.current_token()
        if token.type == TokenType.NUMBER:
            self.consume()
            return NumberLiteral(token.value)
        elif token.type == TokenType.IDENTIFIER:
            self.consume()
            return Variable(token.value)
        elif token.type == TokenType.LPAREN:
            self.consume()  # Consume '('
            expr = self.parse_expr()
            self.consume()  # Consume ')'
            return expr
        elif token.type == TokenType.LBRACKET:
            return self.parse_list_expr()
        else:
            raise SyntaxError(f"Unexpected token: {token.value} in {example_code}")

    def parse_binary_expr(self, left, precedence):
        op = self.consume()
        right = self.parse_expr(precedence)
        return BinaryExpr(left, right, op)

    def get_precedence(self, token):
        precedences = {
            TokenType.PLUS: 1,
            TokenType.MINUS: 1,
            TokenType.MULTIPLY: 2,
            TokenType.DIVIDE: 2,
        }
        return precedences.get(token.type, 0)

    def parse_function_decl(self, name) -> FuncDecl:
        pass

    def parse_var_decl(self, name) -> VarDecl:
        expr = self.parse_expr()
        return VarDecl(name, expr)

    def parse_declaration(self) -> Decl:
        name = self.current_token().value
        if self.peek(1).type == TokenType.ASSIGN and self.peek(2).type == TokenType.PIPE:
            return self.parse_function_decl(name)
        #matched variable decleration like let myvarname = 10 + 10 consume myvarname and =
        elif self.peek(1).type == TokenType.ASSIGN:
            self.consume()
            self.consume()
            return self.parse_var_decl(name)


    def parse_statement(self) -> Statement:
        #matched let myname consume the let
        if self.current_token().type == TokenType.LET and self.peek(1).type == TokenType.IDENTIFIER:
            self.consume()
            return self.parse_declaration()
        #not a declaration so something I need to evaluate
        else:
            return self.parse_expr()

    # [1, 1+2, 3, variable]
    def parse_list_expr(self) -> ListExpr:
        exprs = []
        while self.current_token().type != TokenType.RBRACKET:
            self.consume()
            expr = self.parse_expr()
            exprs.append(expr)
            print(self.current_token())
            #here should be , now
        self.consume()
        return ListExpr(exprs)

# Example usage
example_code = '''
let a = [(1+b)*3,2,3]
'''

tokens = tokenize(example_code)
parser = Parser(tokens)
statements = parser.parse()
print(statements)
program = Program(statements)
program.execute()
