import sys

from baumlang.parser import run
from baumlang.scanner import Token, tokenize

def tokenize_file(file_path):
    """
    Read content from a file and tokenize it.

    :param file_path: Path to the file to be tokenized
    :return: List of tokens
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return tokenize(content)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except IOError:
        print(f"Error: Unable to read file '{file_path}'.")
        return []

def __main__(*args):
    if len(args) == 0:
        print("Usage: script.py <filename>")
        return

    filename = args[0]

    try:
        with open(filename, 'r') as file:
            content = file.read()
            run(content)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    __main__(*sys.argv[1:])
