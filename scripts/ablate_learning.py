#!/usr/bin/env python3
"""Retrain current learning tasks with feature blocks removed."""

import sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config, write_json
from coptic_examples.lexicon import load
from coptic_examples.inclusion import (
    EntryFeatures,
    COLUMNS_INCLUSION,
    SEMANTIC,
    ALIGNMENT,
)
from coptic_examples.learning import fit_inclusion, fit_score, predict, augment
from coptic_examples.ablation import expected_top_metric
from coptic_examples.features import UD

cfg = Config()
lex = load(cfg.lexicon)
report = json.loads((cfg.reports / "learning/comparison.json").read_text())
raw = [json.loads(s) for s in (cfg.derived / "learning_examples.jsonl").open()]
sets = {
    "without_POS_alignment": set(ALIGNMENT),
    "without_definition_similarity": set(SEMANTIC[:10]),
    "without_punctuation_pronouns": {
        "n_punctuation",
        "n_commas",
        "n_parentheses",
        "pronoun_ratio",
        "bound_personal_pronoun_ratio",
        "independent_personal_pronoun_ratio",
    },
    "without_length": {
        "sent_len",
        "n_tokens",
        "n_chars",
        "avg_word_length",
        "max_word_length",
        "min_word_length",
        "trans_len",
    },
    "without_UD": set(UD)
    | set(ALIGNMENT)
    | {
        "dependency_target_found",
        "filled_argument_roles",
        "target_descriptor_count",
        "target_in_main_clause",
        "frame_log_frequency",
        "frame_probability",
        "collocation_log_mean",
        "collocation_log_max",
    },
}
results = []
for d in report["results"]:
    dialect = d["dialect"]
    data = [r for r in raw if r["dialect"] == dialect]
    ef = EntryFeatures(cfg, lex, dialect)
    for r in data:
        r["x"] = ef.vector(r["candidate"], lex[r["tla"]])
    fam = sorted({r["family"] for r in data})
    np.random.default_rng(42).shuffle(fam)
    folds = {f: i % 5 for i, f in enumerate(fam)}
    memo = {}
    predictions = defaultdict(list)
    for fold in range(5):
        val = [r for r in data if folds[r["family"]] == fold]
        held = {r["text_key"] for r in val}
        train = [
            r for r in data if folds[r["family"]] != fold and r["text_key"] not in held
        ]
        if not val or len({r["y"] for r in train}) < 2:
            continue
        extras = {}
        for arch, choice in d["choices"].items():
            s = choice["setting"]
            ratio = float(s.rsplit("_", 1)[1]) if s != "none" else 0
            kind = (
                "hard"
                if s.startswith("hard")
                else "unrelated" if s.startswith("unrelated") else "mixed"
            )
            extras[arch] = augment(train, ratio, kind, ef, lex, memo)
        for name, removed in sets.items():
            indices = [i for i, c in enumerate(COLUMNS_INCLUSION) if c in removed]

            def mask(rows):
                result = []
                for r in rows:
                    x = list(r["x"])
                    for i in indices:
                        x[i] = float("nan")
                    result.append(dict(r, x=x))
                return result

            tr = mask(train)
            va = mask(val)
            a = fit_inclusion(tr + mask(extras["two_models"]))
            b = fit_score(tr, False)
            c = fit_score(tr + mask(extras["joint"]), True)
            for arch, inc, rank in [
                ("two_models", predict(a, va), predict(b, va)),
                ("joint", None, predict(c, va)),
            ]:
                if inc is None:
                    inc = 1 / (1 + np.exp(-np.clip(rank, -30, 30)))
                predictions[(arch, name)].extend(
                    (r, float(p), float(s)) for r, p, s in zip(val, inc, rank)
                )
        print(dialect, "ablation fold", fold, flush=True)
    summaries = []
    for (arch, name), values in predictions.items():
        y = [r["y"] > 0 for r, p, s in values]
        inc = [p for r, p, s in values]
        groups = defaultdict(list)
        for r, p, s in values:
            for rank in r["ranks"]:
                groups[(r["tla"], rank["source"])].append((rank["grade"], s))
        ranks = {}
        for source in ["tab", "crum"]:
            scores = [
                expected_top_metric([a for a, b in rows], [b for a, b in rows], 6)
                for (t, s), rows in groups.items()
                if s == source and len({a for a, b in rows}) > 1
            ]
            ranks[source] = float(np.mean(scores)) if scores else None
        summaries.append(
            dict(
                architecture=arch,
                setup=name,
                removed=sorted(sets[name] & set(COLUMNS_INCLUSION)),
                auc=float(roc_auc_score(y, inc)),
                macro_f1=float(f1_score(y, np.asarray(inc) > 0.5, average="macro")),
                ranking=ranks,
            )
        )
    results.append(dict(dialect=dialect, summaries=summaries))
    ef.close()
write_json(
    cfg.reports / "learning/ablations.json",
    dict(
        results=results,
        protocol="Same folds, augmentation selections and thresholds as the current model comparison. Removed feature columns are masked in training and evaluation; each model is retrained.",
    ),
)
