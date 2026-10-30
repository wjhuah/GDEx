#!/usr/bin/env python3
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coptic_examples.config import Config
from coptic_examples.crum_audit import run
from export_crum_review import export
if __name__=='__main__':
    config=Config()
    print(run(config))
    print(export(config))
