#!/usr/bin/env python3
"""Local entry point; also installable as coptic-examples."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from coptic_examples.cli import main

if __name__ == "__main__":
    main()
