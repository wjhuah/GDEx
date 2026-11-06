#!/usr/bin/env python3
"""Compare class balancing with class-and-entry balancing on identical folds."""

import sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config, write_json
from coptic_examples.lexicon import load
from coptic_examples.inclusion import EntryFeatures
from coptic_examples.learning import fit_inclusion, fit_score, predict, augment

cfg = Config()
lex = load(cfg.lexicon)
report = json.loads((cfg.reports / "learning/comparison.json").read_text())
out = []
for d in report["results"]:
    data = [
        json.loads(s)
        for s in (cfg.derived / ("model_rows_" + d["dialect"] + ".jsonl")).open()
    ]
    for r in data:
        r["x"] = [np.nan if v is None else v for v in r["x"]]
    ef = EntryFeatures(cfg, lex, d["dialect"])
    family = sorted({r["family"] for r in data})
    np.random.default_rng(42).shuffle(family)
    folds = {f: i % 5 for i, f in enumerate(family)}
    memo = {}
    pred = defaultdict(list)
    for fold in range(5):
        val = [r for r in data if folds[r["family"]] == fold]
        held = {r["text_key"] for r in val}
        train = [
            r for r in data if folds[r["family"]] != fold and r["text_key"] not in held
        ]
        if not val or len({r["y"] for r in train}) < 2:
            continue
        for arch, c in d["choices"].items():
            s = c["setting"]
            ratio = 0 if s == "none" else float(s.rsplit("_", 1)[1])
            kind = (
                "hard"
                if s.startswith("hard")
                else "unrelated" if s.startswith("unrelated") else "mixed"
            )
            extra = augment(train, ratio, kind, ef, lex, memo)
            model = (
                fit_inclusion(train + extra, balance="entry_class")
                if arch == "two_models"
                else fit_score(train + extra, True, balance="entry_class")
            )
            p = predict(model, val)
            if arch == "joint":
                p = 1 / (1 + np.exp(-np.clip(p, -30, 30)))
            pred[arch].extend((r["y"] > 0, float(v)) for r, v in zip(val, p))
        print(d["dialect"], fold, flush=True)
    for arch, values in pred.items():
        y, p = map(np.asarray, zip(*values))
        baseline = next(
            r
            for r in d["summaries"]
            if r["architecture"] == arch
            and r["setting"] == d["choices"][arch]["setting"]
        )
        out.append(
            dict(
                dialect=d["dialect"],
                architecture=arch,
                class_balanced_auc=baseline["auc"],
                class_balanced_f1=baseline["macro_f1"],
                entry_class_balanced_auc=float(roc_auc_score(y, p)),
                entry_class_balanced_f1=float(f1_score(y, p > 0.5, average="macro")),
            )
        )
    ef.close()
write_json(cfg.reports / "learning/negative_weight_comparison.json", out)
