import re
from _ast import Expression
from enum import Enum, auto
from typing import TypeVar, List, Union, Callable


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
    AND = auto()
    OR = auto()
    TRUE = auto()
    FALSE = auto()

class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __repr__(self):
        return f"({self.type}, {self.value})"

    def is_operator(self):
        return (self.type == TokenType.PLUS or
                self.type == TokenType.MINUS or
                self.type == TokenType.MULTIPLY or
                self.type == TokenType.DIVIDE or
                self.type == TokenType.EQUALS or
                self.type == TokenType.GREATER or
                self.type == TokenType.NOT or
                self.type == TokenType.LESS or
                self.type == TokenType.AND or
                self.type == TokenType.OR)

def tokenize(code):
    tokens = []
    keywords = {
        'let': TokenType.LET,
        'if': TokenType.IF,
        'else': TokenType.ELSE,
        'true': TokenType.TRUE,
        'false': TokenType.FALSE
    }

    token_specifications = [
        ('NUMBER', r'\d+(\.\d*)?'),
        ('STRING', r'"[^"]*"'),
        ('TRUE', r'true'),
        ('FALSE', r'false'),
        ('IDENTIFIER', r'[a-zA-Z_]\w*'),
        ('AND', r'&&'),
        ('OR', r'\|\|'),
        ('EQUALS', r'=='),
        ('ASSIGN', r'='),
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

class FuncCall(Expr):
    def __init__(self, name, args):
        self.name = name
        self.args = args

class IfExpr(Expr):
    def __init__(self, condition, thn, els):
        self.condition = condition
        self.thn = thn
        self.els = els

T = TypeVar('T')

class Program:
    def __init__(self, stmts):
        self.statements = stmts
        self.vars = {}
        self.funcs = {}

    def execute(self):
        for stmt in self.statements:
            if isinstance(stmt, Expr):
                self.eval(stmt)
            elif isinstance(stmt, VarDecl):
                self.vars[stmt.name] = self.eval_var(stmt)
            elif isinstance(stmt, FuncDecl):
                self.funcs[stmt.name] = stmt

    def eval(self, expr: Expr) -> Union[int, float, List, Expr]:
        if isinstance(expr, NumberLiteral):
            return expr.value
        if isinstance(expr, BooleanLiteral):
            return expr.value
        elif isinstance(expr, Variable):
            variable = self.vars.get(expr.name)
            if variable is None:
                variable = self.funcs.get(expr.name)
            return variable  # Default to 0 if variable not found
        elif isinstance(expr, NotExpr):
            return not self.eval(expr.expr)
        elif isinstance(expr, IfExpr):
            cond_result  = self.eval(expr.condition)
            if cond_result:
                return self.eval(expr.thn)
            else:
                if expr.els is not None:
                    return self.eval(expr.els)
                else:
                    return []
        elif isinstance(expr, BinaryExpr):
            left = self.eval(expr.lhs)
            right = self.eval(expr.rhs)
            if expr.operator.type == TokenType.PLUS:
                return left + right
            elif expr.operator.type == TokenType.MINUS:
                return left - right
            elif expr.operator.type == TokenType.OR:
                return left or right
            elif expr.operator.type == TokenType.AND:
                return left and right
            elif expr.operator.type == TokenType.MULTIPLY:
                return left * right
            elif expr.operator.type == TokenType.DIVIDE:
                return left / right
            elif expr.operator.type == TokenType.EQUALS:
                return left == right
            elif expr.operator.type == TokenType.LESS:
                return left < right
            elif expr.operator.type == TokenType.GREATER:
                return left > right
        elif isinstance(expr, ListExpr):
            return [self.eval(e) for e in expr.elements]
        elif isinstance(expr, FuncCall):
            if expr.name.name == "print":
                print("Printing:", [self.eval(arg) for arg in expr.args])
                return []
            if expr.name.name == "len":
                return len(self.eval(expr.args[0]))
            if expr.name.name == "concat":
                result = []
                for arg in expr.args:
                    evaluated_arg = self.eval(arg)
                    if isinstance(evaluated_arg, list):
                        result.extend(evaluated_arg)
                    else:
                        result.append(evaluated_arg)
                return result
            if expr.name.name == "head":
                return self.eval(expr.args[0])[0]
            if expr.name.name == "tail":
                return self.eval(expr.args[0])[1:]
            func = self.funcs.get(expr.name.name)
            if func is None:
                func = self.vars.get(expr.name.name)
            args = [self.eval(arg) for arg in expr.args]
            return self.eval_func(func, args)
        else:
            raise ValueError(f"Unsupported expression type: {type(expr)} {expr}")

    def eval_func(self, func: FuncDecl, args: List) -> Union[int, float, List]:
        if len(func.args) != len(args):

            raise ValueError(f"Function {func.name} expects {len(func.args)} arguments, but got {len(args)}")

        # Create a new scope for the function
        old_vars = self.vars.copy()
        for param, arg in zip(func.args, args):
            self.vars[param] = arg

        result = self.eval(func.body)

        # Restore the old scope
        self.vars = old_vars

        return result

    def eval_var(self, stmt: VarDecl) -> Union[int, float, List]:
        return self.eval(stmt.expr)


class ListExpr(Expr):
    def __init__(self, elements):
        self.elements = elements

class NotExpr(Expression):
    def __init__(self, expr):
        self.expr = expr

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


class BooleanLiteral(Expr):
    def __init__(self, value):
        self.value = value


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

    def parse_function_call(self, func)-> FuncCall:
        args = []
        self.consume()  # Consume '('
        while self.current_token().type != TokenType.RPAREN:
            args.append(self.parse_expr())
            if self.current_token().type == TokenType.COMMA:
                self.consume()
        self.consume()  # Consume ')'
        return FuncCall(func, args)

    def parse_expr(self, precedence=0) -> Expr:
        if self.current_token().type == TokenType.IF:
            return self.parse_if_expr()
        if self.current_token().type == TokenType.LBRACKET:
            return self.parse_list_expr()
        if self.current_token().type == TokenType.NOT:
            self.consume()
            return NotExpr(self.parse_expr())
        left = self.parse_primary()
        while self.tokens_left():
            if self.current_token().type == TokenType.LPAREN:
                # This is a function call
                left = self.parse_function_call(left)
            elif self.current_token().is_operator() and self.get_precedence(self.current_token()) >= precedence:
                left = self.parse_binary_expr(left, self.get_precedence(self.current_token()))
            else:
                break

        return left

    def parse_primary(self):
        token = self.current_token()
        if token.type == TokenType.NUMBER:
            self.consume()
            return NumberLiteral(token.value)
        elif token.type == TokenType.TRUE:
            self.consume()
            return BooleanLiteral(True)
        elif token.type == TokenType.FALSE:
            self.consume()
            return BooleanLiteral(False)
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
            raise SyntaxError(f"Unexpected token: {self.current_token(), self.peek(1), self.peek(2)}")

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
            TokenType.EQUALS: 3,
            TokenType.AND: 0,
            TokenType.OR: 0,
            TokenType.LESS: 0,
            TokenType.GREATER: 0,

        }
        return precedences.get(token.type, 0)

    def parse_function_decl(self, name) -> FuncDecl:
        self.consume()# name
        self.consume()# =
        self.consume()# start |
        params = self.parse_parameters()
        self.consume() # end |
        self.consume() # start {
        body = self.parse_expr()
        self.consume() # end }
        return FuncDecl(name, params, body)

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
        if self.current_token().type == TokenType.LBRACKET and self.peek(1).type == TokenType.RBRACKET:
            self.consume()
            self.consume()
            return ListExpr(exprs)

        while self.current_token().type != TokenType.RBRACKET:
            self.consume()
            expr = self.parse_expr()
            exprs.append(expr)
            #here should be , now
        self.consume()

        return ListExpr(exprs)

    def parse_parameters(self) -> [str]:
        params = []
        while self.current_token().type != TokenType.PIPE:
            if self.current_token().type == TokenType.IDENTIFIER:
                params.append(self.current_token().value)
            self.consume()

        return params

    def parse_if_expr(self) -> IfExpr:
        self.consume()
        ifexpr = self.parse_expr()
        self.consume()
        thn = self.parse_expr()
        self.consume()
        elseexpr = None
        if self.current_token().type == TokenType.ELSE:
            self.consume()
            self.consume()
            elseexpr = self.parse_expr()
            self.consume()

        expr = IfExpr(ifexpr, thn, elseexpr)
        return expr
#
# example_code = '''
# let add1 = |a| {
#     a + 1
# }
# let map = |f, lst| {
#     if len(lst) == 0 {
#         []
#     } else {
#         concat(f(head(lst)), map(f, tail(lst)))
#     }
# }
#
# let reduce = |f, lst, initial| {
#     if len(lst) == 0 {
#         initial
#     } else {
#         reduce(f, tail(lst), f(initial, head(lst)))
#     }
# }
#
# let add = |a, b| {
#     a + b
# }
#
# let sum = |lst| {
#     reduce(add, lst, 0)
# }
#
# print(map(add1, [1,2])) # should print [2,3]
# print(sum([1, 2, 3, 4, 5]))  # Should print 15
#
# '''
# try:
#     print("======================")
#     print(example_code)
#     print("======================")
#
#     tokens = tokenize(example_code)
#     parser = Parser(tokens)
#     statements = parser.parse()
#     # for stmt in statements:
#     #     if isinstance(stmt, Expr):
#     #         print("EXECUTE", stmt)
#     #     else:
#     #         print("DECLS", stmt)
#     program = Program(statements)
#     program.execute()
# except Exception as e:
#     raise
#
