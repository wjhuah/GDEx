#!/usr/bin/env python3
"""Fill the optional BGE cache for current definitions and source examples."""

import sys, json, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config
from coptic_examples.lexicon import load
from coptic_examples.embeddings import Embeddings

parser = argparse.ArgumentParser()
parser.add_argument("--sentences", action="store_true")
args = parser.parse_args()
cfg = Config()
embedding = Embeddings(cfg.vectors)
embedding.encode([e.gloss for e in load(cfg.lexicon).values() if e.gloss], "definition")
if args.sentences:
    rows = [json.loads(s) for s in (cfg.derived / "ablation_rows.jsonl").open()]
    embedding.encode(
        [r["candidate"]["translation"] for r in rows if r["candidate"]["translation"]],
        "sentence",
    )
