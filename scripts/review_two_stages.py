"""Reproduce the November review from frozen features and out-of-fold predictions."""

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config
from coptic_examples.ablation import expected_top_metric

cfg = Config()
lines = ["# November 12 model review", ""]
sources = {
    p.name: p.read_bytes() for p in (cfg.source / "Crum_plain_text").glob("*.txt")
}
hashes = {k: hashlib.sha256(v).hexdigest() for k, v in sources.items()}
markers = sum(v.lower().count(b"{bible:") for v in sources.values())
ids, positions, statuses = set(), set(), Counter()
for line in (cfg.derived / "complete_citations/citations.jsonl").open():
    r = json.loads(line)
    key = (r["file"], r["byte_start"])
    assert r["id"] not in ids and key not in positions
    ids.add(r["id"])
    positions.add(key)
    raw = r["raw"].encode()
    assert hashes[key[0]] == r["file_sha256"]
    assert (
        sources[key[0]][key[1] : key[1] + len(raw)] == raw and b"{bible:" in raw.lower()
    )
    statuses[r["training_status"]] += 1
assert markers == len(ids)
lines += [
    f"Across {len(sources):,} transcription files, {markers:,} explicit Bible markers match {len(ids):,} unique ledger records. Every record matches the source file hash and the text at its byte offset. Repeated references at distinct positions count separately.",
    "",
    "| Citation status | Occurrences |",
    "| --- | ---: |",
]
lines += [f"| {s.replace('_',' ')} | {n:,} |" for s, n in statuses.most_common()]
summary = json.loads((cfg.reports / "forest/comparison.json").read_text())
for d in summary["results"]:
    dialect = d["dialect"]
    oof = {
        (r["tla"], r["id"]): r
        for r in json.loads((cfg.reports / f"forest/oof_{dialect}.json").read_text())
    }
    held = defaultdict(set)
    for r in oof.values():
        held[r["fold"]].add(r["text_key"])
    obs = []
    pairs = set()
    crum = positives = 0
    for line in (cfg.derived / f"model_rows_{dialect}.jsonl").open():
        r = json.loads(line)
        key = (r["tla"], r["text_key"])
        assert key not in pairs
        pairs.add(key)
        positives += r["y"] > 0
        crum += r["y"] > 0 and "crum" in r["sources"]
        pred = oof[(r["tla"], r["candidate"]["id"])]
        assert pred["text_key"] == r["text_key"] and pred["y"] == r["y"]
        if r["y"] > 0:
            for grade in r["ranks"]:
                obs.append(
                    (
                        r["x"],
                        grade["grade"],
                        pred["fold"],
                        key[1],
                        key[0],
                        grade["source"],
                        pred["rank"],
                    )
                )
    assert len(pairs) == len(oof) == d["rows"]
    lines += [
        "",
        f"## {dialect.title()}",
        "",
        f"{len(pairs):,} unique entry–text pairs include {positives:,} positives ({crum:,} with Crum provenance) and {len(pairs)-positives:,} negatives. Final classification training adds {d['final_augmentation']:,} generated negatives.",
        "",
        "### Inclusion classification",
        "",
        "Both classifiers used the same five spelling-family folds, train-only augmentation and duplicate-text purge. Precision, recall and macro F1 use a fixed 0.5 threshold.",
        "",
        "| Classifier | ROC AUC | Average precision | Precision | Recall | Macro F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, label in [("lgb", "LightGBM GBDT"), ("rf", "Random forest")]:
        m = d["classification"][name]["fixed_05"]
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(
                f"{m[k]:.4f}"
                for k in ["auc", "average_precision", "precision", "recall", "macro_f1"]
            )
            + " |"
        )
    x = np.asarray([o[0] for o in obs], float)
    y = np.array([o[1] for o in obs])
    predictions = np.zeros(len(obs))

    def forest():
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestRegressor(
                n_estimators=200,
                max_depth=16,
                min_samples_leaf=5,
                n_jobs=4,
                random_state=42,
            ),
        )

    for fold in range(5):
        train = [
            i for i, o in enumerate(obs) if o[2] != fold and o[3] not in held[fold]
        ]
        val = [i for i, o in enumerate(obs) if o[2] == fold]
        if val:
            predictions[val] = forest().fit(x[train], y[train]).predict(x[val])
    lines += [
        "",
        "### Preference ranking",
        "",
        "Both regressors used graded positive examples on the same folds. NDCG@6 is averaged over entries with at least two distinct grades within a source. This isolates ordering before the inclusion threshold and display filters.",
        "",
        "| Ranker | Source | Entry groups | NDCG@6 |",
        "| --- | --- | ---: | ---: |",
    ]
    for label, pr in [
        ("LightGBM GBDT", [o[6] for o in obs]),
        ("Random forest", predictions),
    ]:
        groups = defaultdict(list)
        for o, p in zip(obs, pr):
            groups[(o[4], o[5])].append((o[1], p))
        for source in ["tab", "crum"]:
            scores = [
                expected_top_metric(*zip(*v), 6)
                for (_, s), v in groups.items()
                if s == source and len(set(a for a, b in v)) > 1
            ]
            lines.append(
                f"| {label} | {source.upper()} | {len(scores)} | {np.mean(scores):.4f} |"
            )
    cols = json.loads((cfg.derived / "feature_schema.json").read_text())["columns"]
    imp = dict(zip(cols, forest().fit(x, y).steps[-1][1].feature_importances_))
    ranks = {
        n: {r["feature"]: r["gain_share"] for r in d["lightgbm_importance"][n]}
        for n in ["inclusion_v2", "ranking_v2"]
    }
    lines += [
        "",
        "### Feature importance",
        "",
        "Percentages show the share of training impurity reduction for random forests and split gain for GBDT. Correlated features can share credit. Rows follow random forest inclusion importance.",
        "",
        "| Feature | RF inclusion | GBDT inclusion | RF ranking | GBDT ranking |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for f in d["forest_importance"]:
        k = f["feature"]
        lines.append(
            f"| {k} | {f['mdi']:.2%} | {ranks['inclusion_v2'][k]:.2%} | {imp[k]:.2%} | {ranks['ranking_v2'][k]:.2%} |"
        )
path = cfg.reports / "forest/review_2026-11-12.md"
path.write_text("\n".join(lines) + "\n")
