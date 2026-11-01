import sys
from petseg.cli import main

if __name__ == "__main__":
    main(["predict", *sys.argv[1:]])
