import sys

from semantic_speech.cli import main


if __name__ == "__main__":
    main(["verify", *sys.argv[1:]])
