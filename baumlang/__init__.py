# Assuming these are defined in your other files
import argparse
import sys
from typing import List

from baumlang.runtime import tokenize, Parser, Expr, Program


def read_file(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read()

def parse_and_execute(code: str, debug: bool = False) -> None:
    tokens = tokenize(code)
    parser = Parser(tokens)
    statements = parser.parse()

    if debug:
        print("Abstract Syntax Tree:")
        for stmt in statements:
            print(f"EXPRESSION: {stmt}")
        print("\nExecuting program:\n")

    program = Program(statements)
    program.execute()

def main(args: List[str]) -> None:
    parser = argparse.ArgumentParser(description="Execute code files or run in interactive mode.")
    parser.add_argument('files', nargs='*', help='Input files to execute')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to print AST')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')

    args = parser.parse_args(args)

    if args.interactive:
        print("Interactive mode. Type your code and press Enter. Type 'exit' to quit.")
        while True:
            try:
                code = input(">>> ")
                if code.lower() == 'exit':
                    break
                parse_and_execute(code, args.debug)
            except Exception as e:
                print(f"Error: {e}")
    elif args.files:
        for file_path in args.files:
            try:
                code = read_file(file_path)
                print(f"Executing file: {file_path}")
                parse_and_execute(code, args.debug)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    else:
        print("No input files provided. Use --interactive for interactive mode or provide file names.")

if __name__ == "__main__":
    main(sys.argv[1:])