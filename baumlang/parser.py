from enum import Enum
from dataclasses import dataclass
from typing import Union, Optional


class TokenType(Enum):
    Num = 0
    Boolean = 1
    Struct = 2
    RightArr = 3
    Assign = 4
    Equal = 5
    Greater = 6
    Lower = 7
    Bigger = 8
    Plus = 9
    Minus = 10
    Mult = 11
    Div = 12
    LPar = 13
    RPar = 14
    LBrack = 15
    RBrack = 16
    Comma = 17
    Append = 18
    Val = 19
    Print = 20
    Head = 21
    Tail = 22
    Name = 23
@dataclass
class Token:
    tokenType: TokenType
    value: Optional[Union[str, float, int]]

class Scanner:
    @staticmethod
    def scan_tokens(filename: str):
        tokens = []
        with open(filename, 'r') as f:
            content = f.read()
            i = 0
            line = 0
            while i < len(content):
                char = content[i]

                if char.isdigit() or (char == '.' and i + 1 < len(content) and content[i + 1].isdigit()):
                    # Handle numbers (integers and floats)
                    num_str = ''
                    while i < len(content) and (content[i].isdigit() or content[i] == '.'):
                        num_str += content[i]
                        i += 1
                    tokens.append(Token(TokenType.Num, float(num_str) if '.' in num_str else int(num_str)))
                    continue

                elif char.isalpha():
                    # Handle keywords and identifiers
                    identifier = ''
                    while i < len(content) and (content[i].isalpha() or content[i].isdigit() or content[i] == '_'):
                        identifier += content[i]
                        i += 1

                    # Keywords handling
                    if identifier == "true" or identifier == "false":
                        tokens.append(Token(TokenType.Boolean, identifier == "true"))
                    elif identifier == "struct":
                        tokens.append(Token(TokenType.Struct, identifier))
                    elif identifier == "++":
                        tokens.append(Token(TokenType.Append, identifier))
                    elif identifier == "->":
                        tokens.append(Token(TokenType.RightArr, identifier))
                    elif identifier == "val":
                        tokens.append(Token(TokenType.Val, identifier))
                    elif identifier == "print":
                        tokens.append(Token(TokenType.Print, identifier))
                    elif identifier == "head":
                        tokens.append(Token(TokenType.Head, identifier))
                    elif identifier == "tail":
                        tokens.append(Token(TokenType.Tail, identifier))
                    else:
                        tokens.append(Token(TokenType.Name, identifier))
                    continue

                elif char == '=':
                    tokens.append(Token(TokenType.Assign, char))

                elif char == '>':
                    tokens.append(Token(TokenType.Greater, char))

                elif char == '<':
                    tokens.append(Token(TokenType.Lower, char))

                elif char == '+':
                    tokens.append(Token(TokenType.Plus, char))

                elif char == '-':
                    tokens.append(Token(TokenType.Minus, char))

                elif char == '*':
                    tokens.append(Token(TokenType.Mult, char))

                elif char == '/':
                    tokens.append(Token(TokenType.Div, char))

                elif char == '(':
                    tokens.append(Token(TokenType.LPar, char))

                elif char == ')':
                    tokens.append(Token(TokenType.RPar, char))

                elif char == '[':
                    tokens.append(Token(TokenType.LBrack, char))

                elif char == ']':
                    tokens.append(Token(TokenType.RBrack, char))

                elif char == ',':
                    tokens.append(Token(TokenType.Comma, char))

                elif char == '@':
                    tokens.append(Token(TokenType.Append, char))

                elif char.isspace():
                    i += 1
                    continue

                elif char == '\n':
                    # Skip newlines, but you could also handle line numbers here if needed
                    i += 1
                    line += 1
                    continue

                else:
                    print(tokens)
                    raise ValueError(f"Unknown character: {char} in line {line}")

                i += 1

        return tokens
