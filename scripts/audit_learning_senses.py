#!/usr/bin/env python3
"""Check the five spelling groups against the current dictionary and held-out models."""

import sys, json, copy
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config, write_json
from coptic_examples.lexicon import load
from coptic_examples.corpus import Corpus
from coptic_examples.learning import augment, fit_inclusion, fit_score, predict
from coptic_examples.inclusion import EntryFeatures
from coptic_examples.sense import native_candidates, align_candidate

cfg = Config()
lex = load(cfg.lexicon)
corpus = Corpus(cfg)
report = json.loads((cfg.reports / "learning/comparison.json").read_text())
rows = [json.loads(s) for s in (cfg.derived / "learning_examples.jsonl").open()]
spellings = {"ϩⲟⲟⲩ", "ⲧⲱⲙⲧ", "ⲉϩⲉ", "ⲡⲉ", "ⲉ"}
entries = [
    e for e in lex.values() if e.lemma in spellings or e.tla in {"C992", "C993", "C998"}
]
ids = {e.tla for e in entries}
inventory = []
heldout = []
contrasts = []
failures = []
for result in report["results"]:
    dialect = result["dialect"]
    data = [r for r in rows if r["dialect"] == dialect]
    by = defaultdict(list)
    for r in data:
        by[r["tla"]].append(r)
    for entry in entries:
        rs = by[entry.tla]
        inventory.append(
            dict(
                tla=entry.tla,
                lemma=entry.lemma,
                dialect=dialect,
                gloss=entry.gloss,
                source=entry.gloss_source,
                forms=entry.forms,
                candidates=len(native_candidates(cfg, corpus, entry, dialect, True)),
                positive=sum(r["y"] > 0 for r in rs),
                negative=sum(r["y"] < 0 for r in rs),
                ordered=sum(bool(r["ranks"]) for r in rs),
            )
        )
    oof = json.loads((cfg.reports / ("learning/oof_" + dialect + ".json")).read_text())
    for architecture, choice in result["choices"].items():
        selected = [
            r
            for r in oof
            if r["architecture"] == architecture
            and r["setting"] == choice["setting"]
            and r["tla"] in ids
        ]
        for tla in sorted({r["tla"] for r in selected}):
            rs = [r for r in selected if r["tla"] == tla]
            p = [r for r in rs if r["y"] > 0]
            n = [r for r in rs if r["y"] < 0]
            heldout.append(
                dict(
                    dialect=dialect,
                    architecture=architecture,
                    tla=tla,
                    positive=len(p),
                    negative=len(n),
                    positive_recall=(
                        sum(r["inclusion"] > 0.5 for r in p) / len(p) if p else None
                    ),
                    negative_recall=(
                        sum(r["inclusion"] <= 0.5 for r in n) / len(n) if n else None
                    ),
                )
            )
    del oof
    ef = EntryFeatures(cfg, lex, dialect)
    for r in data:
        r["x"] = ef.vector(r["candidate"], lex[r["tla"]])
    fam = sorted({r["family"] for r in data})
    np.random.default_rng(42).shuffle(fam)
    folds = {f: i % 5 for i, f in enumerate(fam)}
    known = {(r["tla"], r["text_key"]) for r in data if r["y"] > 0}
    totals = defaultdict(Counter)
    memo = {}
    for fold in range(5):
        val = [r for r in data if folds[r["family"]] == fold]
        held = {r["text_key"] for r in val}
        train = [
            r for r in data if folds[r["family"]] != fold and r["text_key"] not in held
        ]
        probes = []
        for r in val:
            if r["tla"] not in ids or r["y"] < 0:
                continue
            for rival in ef.rivals(lex[r["tla"]]):
                if rival.tla not in ids or (rival.tla, r["text_key"]) in known:
                    continue
                c = copy.deepcopy(r["candidate"])
                ii = align_candidate(c, rival, True)
                if not ii:
                    continue
                source = align_candidate(copy.deepcopy(c), lex[r["tla"]], True)
                if not set(ii) & set(source):
                    continue
                probes.append((r, dict(r, tla=rival.tla, x=ef.vector(c, rival))))
        if not probes or len({r["y"] for r in train}) < 2:
            continue
        for architecture, choice in result["choices"].items():
            setting = choice["setting"]
            ratio = float(setting.rsplit("_", 1)[1]) if setting != "none" else 0.0
            kind = (
                "hard"
                if setting.startswith("hard")
                else "unrelated" if setting.startswith("unrelated") else "mixed"
            )
            extra = augment(train, ratio, kind, ef, lex, memo)
            model = (
                fit_inclusion(train + extra)
                if architecture == "two_models"
                else fit_score(train + extra, True)
            )
            a = predict(model, [p[0] for p in probes])
            b = predict(model, [p[1] for p in probes])
            for (own, rival), x, y in zip(probes, a, b):
                pair = " / ".join(sorted([own["tla"], rival["tla"]]))
                out = totals[(architecture, pair)]
                out["total"] += 1
                out["correct"] += int(x > y + 1e-10)
                out["ties"] += int(abs(x - y) <= 1e-10)
                if x <= y + 1e-10:
                    failures.append(
                        dict(
                            dialect=dialect,
                            architecture=architecture,
                            tla=own["tla"],
                            rival_tla=rival["tla"],
                            sentence_id=own["candidate"]["id"],
                            translation=own["candidate"]["translation"],
                            source_score=float(x),
                            rival_score=float(y),
                        )
                    )
        print(dialect, "contrast fold", fold, flush=True)
    for (architecture, pair), counts in totals.items():
        contrasts.append(
            dict(dialect=dialect, architecture=architecture, pair=pair, **counts)
        )
    ef.close()
write_json(
    cfg.reports / "learning/senses.json",
    dict(inventory=inventory, heldout=heldout, contrasts=contrasts, failures=failures),
)
print(
    "sense audit complete", len(inventory), "entries", len(contrasts), "contrast groups"
)
