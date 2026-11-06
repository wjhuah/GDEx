#!/usr/bin/env python3
"""Train and evaluate token sense contrasts using source-selected examples."""

import sys, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.dialects import Dialects
from coptic_examples.config import Config, write_json
from coptic_examples.lexicon import load
from coptic_examples.inclusion import EntryFeatures
from coptic_examples.token_senses import dataset, fit, COLUMNS
from coptic_examples.learning import model_fingerprints
import hashlib

cfg = Config()
lex = load(cfg.lexicon)
original = [json.loads(s) for s in (cfg.derived / "learning_examples.jsonl").open()]
reports = []
for dialect in Dialects().data:
    base = [r for r in original if r["dialect"] == dialect]
    features = EntryFeatures(cfg, lex, dialect)
    family = sorted({r["family"] for r in base})
    np.random.default_rng(42).shuffle(family)
    folds = {f: i % 5 for i, f in enumerate(family)}
    pred = []
    contrasts = []
    for fold in range(5):
        val = [r for r in base if folds[r["family"]] == fold]
        held = {r["text_key"] for r in val}
        train = [
            r for r in base if folds[r["family"]] != fold and r["text_key"] not in held
        ]
        tr = dataset(train, features, lex)
        va = dataset(val, features, lex)
        m = fit(tr)
        if not m or not va:
            continue
        p = m.predict(np.asarray([r["x"] for r in va]), num_threads=4)
        pred.extend((r["y"], float(v)) for r, v in zip(va, p))
        by = {(r["tla"], r["text_key"]): float(v) for r, v in zip(va, p)}
        for r in va:
            if r["y"] != 1:
                continue
            for e in features.rivals(lex[r["tla"]]):
                key = (e.tla, r["text_key"])
                if key in by:
                    contrasts.append(
                        dict(
                            tla=r["tla"],
                            rival=e.tla,
                            correct=by[(r["tla"], r["text_key"])] > by[key],
                            sentence_id=r["candidate"]["id"],
                        )
                    )
        print(dialect, "fold", fold, len(tr), len(va), flush=True)
    allrows = dataset(base, features, lex)
    m = fit(allrows)
    fitted = []
    if m:
        full = m.predict(np.asarray([r["x"] for r in allrows]), num_threads=4)
        lookup = {(r["tla"], r["text_key"]): float(p) for r, p in zip(allrows, full)}
        for r in allrows:
            if r["y"] != 1:
                continue
            for e in features.rivals(lex[r["tla"]]):
                key = (e.tla, r["text_key"])
                if key in lookup:
                    fitted.append(
                        dict(
                            tla=r["tla"],
                            rival=e.tla,
                            correct=lookup[(r["tla"], r["text_key"])] > lookup[key],
                        )
                    )
    if m:
        m.save_model(str(cfg.work / "models" / dialect / "token_sense.txt"))
    write_json(
        cfg.work / "models" / dialect / "token_sense.json",
        dict(
            **model_fingerprints(cfg, lex),
            manual_sha256=hashlib.sha256(cfg.manual.read_bytes()).hexdigest(),
            columns=COLUMNS,
            available=m is not None,
        ),
    )
    reports.append(
        dict(
            dialect=dialect,
            training_pairs=len(allrows),
            auc=(
                float(roc_auc_score([a for a, b in pred], [b for a, b in pred]))
                if len({a for a, b in pred}) > 1
                else None
            ),
            contrasts=contrasts,
            training_contrasts=fitted,
            columns=COLUMNS,
        )
    )
    features.close()
write_json(cfg.reports / "learning/token_senses.json", reports)
