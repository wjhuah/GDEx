#!/usr/bin/env python3
"""Build per-dialect dependency frequency tables."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config
from coptic_examples.typicality import build

if __name__ == "__main__":
    print(build(Config()))
