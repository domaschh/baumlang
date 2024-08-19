import sys

from baumlang.parser import Scanner


def __main__(*args):
    if len(args) == 0:
        print("Usage: script.py <filename>")
        return

    filename = args[0]

    try:
        tokens = Scanner.scan_tokens(filename)
        for token in tokens:
            print(token)

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    __main__(*sys.argv[1:])
