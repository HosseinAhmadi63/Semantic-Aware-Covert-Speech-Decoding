import sys

from semantic_speech.cli import main


if __name__ == "__main__":
    main(["figures", *sys.argv[1:]])
