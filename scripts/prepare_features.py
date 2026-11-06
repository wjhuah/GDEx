#!/usr/bin/env python3
"""Extract TAB labels, Crum preferences and candidate features."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config
from coptic_examples.ablation import build_rows

if __name__ == "__main__":
    build_rows(Config())
