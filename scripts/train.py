import sys
from petseg.cli import main

if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
