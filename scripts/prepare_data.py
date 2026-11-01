"""Compatibility wrapper; prefer `petseg prepare`."""

import sys
from petseg.cli import main

if __name__ == "__main__":
    main(["prepare", *sys.argv[1:]])
