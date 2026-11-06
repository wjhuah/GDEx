#!/usr/bin/env python3
"""Compare two-stage and joint learning on fixed held-out examples."""

import sys, json, hashlib
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.dialects import Dialects
from coptic_examples.config import Config, write_json
from coptic_examples.lexicon import load
from coptic_examples.learning import (
    build_examples,
    model_fingerprints,
    fit_inclusion,
    fit_score,
    predict,
    augment,
)
from coptic_examples.inclusion import EntryFeatures, COLUMNS_INCLUSION
from coptic_examples.ablation import expected_top_metric

cfg = Config()
lex = load(cfg.lexicon)
allrows = build_examples(cfg)
results = []
settings = [
    ("none", 0.0, "mixed"),
    ("hard_0.5", 0.5, "hard"),
    ("unrelated_0.5", 0.5, "unrelated"),
    ("mixed_0.25", 0.25, "mixed"),
    ("mixed_0.5", 0.5, "mixed"),
    ("mixed_1", 1.0, "mixed"),
    ("mixed_2", 2.0, "mixed"),
]


def measures(rows, inc, rank):
    y = np.asarray([r["y"] > 0 for r in rows])
    signed = np.asarray(inc) > 0.5
    groups = defaultdict(list)
    for r, p in zip(rows, rank):
        for v in r["ranks"]:
            groups[(r["tla"], v["source"])].append((v["grade"], float(p)))
    ranking = {}
    for source in ["tab", "crum"]:
        scores = [
            expected_top_metric([x[0] for x in v], [x[1] for x in v], 6)
            for (t, s), v in groups.items()
            if s == source and len({x[0] for x in v}) > 1
        ]
        ranking[source] = dict(
            groups=len(scores), ndcg_at_6=float(np.mean(scores)) if scores else None
        )
    return dict(
        n=len(rows),
        positive=int(sum(y)),
        negative=int(sum(~y)),
        auc=float(roc_auc_score(y, inc)) if len(set(y)) > 1 else None,
        average_precision=float(average_precision_score(y, inc)) if any(y) else None,
        macro_f1=float(f1_score(y, signed, average="macro", zero_division=0)),
        balanced_accuracy=(
            float(balanced_accuracy_score(y, signed)) if len(set(y)) > 1 else None
        ),
        ranking=ranking,
    )


for dialect in Dialects().data:
    data = [r for r in allrows if r["dialect"] == dialect]
    if not data:
        continue
    ef = EntryFeatures(cfg, lex, dialect)
    for r in data:
        r["x"] = ef.vector(r["candidate"], lex[r["tla"]])
    write_json(
        cfg.derived / "feature_schema.json",
        dict(columns=COLUMNS_INCLUSION, labels={"positive": 1, "negative": -1}),
    )
    with (cfg.derived / ("model_rows_" + dialect + ".jsonl")).open("w") as stream:
        for r in data:
            clean = dict(
                r,
                x=[
                    float(v) if v is not None and np.isfinite(v) else None
                    for v in r["x"]
                ],
            )
            stream.write(json.dumps(clean, ensure_ascii=False, allow_nan=False) + "\n")
    fam = sorted({r["family"] for r in data})
    np.random.default_rng(42).shuffle(fam)
    folds = {f: i % 5 for i, f in enumerate(fam)}
    write_json(
        cfg.derived / ("fold_manifest_" + dialect + ".json"),
        [
            dict(
                tla=r["tla"],
                sentence_id=r["candidate"]["id"],
                text_key=r["text_key"],
                fold=folds[r["family"]],
            )
            for r in data
        ],
    )
    predictions = defaultdict(list)
    counts = []
    memo = {}
    for fold in range(5):
        val = [r for r in data if folds[r["family"]] == fold]
        held = {r["text_key"] for r in val}
        tr = [
            r for r in data if folds[r["family"]] != fold and r["text_key"] not in held
        ]
        if not val or len({r["y"] for r in tr}) < 2:
            continue
        ranker = fit_score(tr, False)
        rank_prediction = predict(ranker, val)
        for setting, ratio, kind in settings:
            extra = augment(tr, ratio, kind, ef, lex, memo)
            assert not any(r["text_key"] in held for r in extra)
            counts.append(
                dict(
                    fold=fold,
                    setting=setting,
                    original=len(tr),
                    positive=sum(r["y"] > 0 for r in tr),
                    negative=sum(r["y"] < 0 for r in tr),
                    added=len(extra),
                    types=dict(Counter(r["sources"][0] for r in extra)),
                )
            )
            classifier = fit_inclusion(tr + extra)
            joint = fit_score(tr + extra, True)
            ci = predict(classifier, val)
            j = predict(joint, val)
            for name, inc, rank in [
                ("two_models", ci, rank_prediction),
                ("joint", 1 / (1 + np.exp(-np.clip(j, -30, 30))), j),
            ]:
                predictions[(name, setting)].extend(
                    dict(row=r, inc=float(a), rank=float(b), fold=fold)
                    for r, a, b in zip(val, inc, rank)
                )
        print(dialect, "fold", fold, "done", flush=True)
    write_json(
        cfg.reports / ("learning/oof_" + dialect + ".json"),
        [
            dict(
                architecture=name,
                setting=setting,
                tla=p["row"]["tla"],
                text_key=p["row"]["text_key"],
                sentence_id=p["row"]["candidate"]["id"],
                y=p["row"]["y"],
                ranks=p["row"]["ranks"],
                fold=p["fold"],
                inclusion=p["inc"],
                rank=p["rank"],
            )
            for (name, setting), values in predictions.items()
            for p in values
        ],
    )
    summaries = []
    for (name, setting), pr in predictions.items():
        summaries.append(
            dict(
                architecture=name,
                setting=setting,
                **measures(
                    [p["row"] for p in pr],
                    [p["inc"] for p in pr],
                    [p["rank"] for p in pr],
                ),
                fold_metrics=[
                    dict(
                        fold=i,
                        **measures(
                            [p["row"] for p in pr if p["fold"] == i],
                            [p["inc"] for p in pr if p["fold"] == i],
                            [p["rank"] for p in pr if p["fold"] == i],
                        ),
                    )
                    for i in sorted({p["fold"] for p in pr})
                ],
            )
        )
    choices = {}
    folder = cfg.work / "models" / dialect
    folder.mkdir(parents=True, exist_ok=True)
    for name in ["two_models", "joint"]:
        eligible = [r for r in summaries if r["architecture"] == name]
        choice = (
            max(eligible, key=lambda r: r["macro_f1"])
            if eligible
            else dict(setting="none")
        )
        setting = choice["setting"]
        ratio, kind = next((r, k) for s, r, k in settings if s == setting)
        extra = augment(data, ratio, kind, ef, lex, memo)
        if name == "two_models":
            model = fit_inclusion(data + extra)
            ranker = fit_score(data, False)
            if model:
                model.save_model(str(folder / "inclusion_v2.txt"))
            if ranker:
                ranker.save_model(str(folder / "ranking_v2.txt"))
        else:
            model = fit_score(data + extra, True)
            if model:
                model.save_model(str(folder / "joint_v2.txt"))
        choices[name] = dict(
            setting=setting,
            training=len(data),
            added=len(extra),
            types=dict(Counter(r["sources"][0] for r in extra)),
        )
    importance = {}
    for name in ["inclusion_v2", "ranking_v2", "joint_v2"]:
        import lightgbm as lgb

        p = folder / (name + ".txt")
        if not p.exists():
            continue
        m = lgb.Booster(model_file=str(p))
        gain = m.feature_importance("gain")
        split = m.feature_importance("split")
        importance[name] = sorted(
            [
                dict(
                    feature=k,
                    gain=float(g),
                    gain_share=float(g / sum(gain)) if sum(gain) else 0.0,
                    split=int(s),
                )
                for k, g, s in zip(m.feature_name(), gain, split)
            ],
            key=lambda r: -r["gain"],
        )
    metadata = dict(
        **model_fingerprints(cfg, lex),
        columns=COLUMNS_INCLUSION,
        choices=choices,
        manual_sha256=hashlib.sha256(cfg.manual.read_bytes()).hexdigest(),
        snapshot_sha256=hashlib.sha256(
            (cfg.work / "cache/cdo_index.json").read_bytes()
        ).hexdigest(),
        label_policy="TAB negatives=-1; inclusion positives=1; TAB priorities=3..1; Crum order=3..1.5",
        default_architecture="two_models",
    )
    write_json(folder / "learning_v2.json", metadata)
    result = dict(
        dialect=dialect,
        examples=len(data),
        sources=dict(Counter(s for r in data for s in r["sources"])),
        counts=counts,
        summaries=summaries,
        choices=choices,
        importance=importance,
    )
    results.append(result)
    ef.close()
write_json(
    cfg.reports / "learning/comparison.json",
    dict(
        version=2,
        protocol="Same spelling-family folds and held-out original examples for every setup; repeated Coptic spans purged; augmentation uses training-fold sentences only. Ratio is requested extra negatives per original positive; available hard-negative pairs bound actual counts. Settings selected using the same folds.",
        results=results,
    ),
)
