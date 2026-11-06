#!/usr/bin/env python3
"""Choose inclusion operating points from development-fold predictions."""

import sys, json
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config, write_json

cfg = Config()
report = json.loads((cfg.reports / "learning/comparison.json").read_text())
output = []
for d in report["results"]:
    oof = json.loads(
        (cfg.reports / ("learning/oof_" + d["dialect"] + ".json")).read_text()
    )
    metadata_path = cfg.work / "models" / d["dialect"] / "learning_v2.json"
    meta = json.loads(metadata_path.read_text())
    meta["thresholds"] = {}
    for arch, choice in d["choices"].items():
        rows = [
            r
            for r in oof
            if r["architecture"] == arch and r["setting"] == choice["setting"]
        ]
        y = np.asarray([r["y"] > 0 for r in rows])
        p = np.asarray([r["inclusion"] for r in rows])
        values = []
        for threshold in np.arange(0.5, 0.995, 0.01):
            z = p > threshold
            values.append(
                dict(
                    threshold=round(float(threshold), 2),
                    macro_f1=float(f1_score(y, z, average="macro")),
                    precision=float(precision_score(y, z, zero_division=0)),
                    recall=float(recall_score(y, z)),
                    balanced_accuracy=float(balanced_accuracy_score(y, z)),
                )
            )
        best = max(values, key=lambda r: r["macro_f1"])
        meta["thresholds"][arch] = best["threshold"]
        output.append(
            dict(
                dialect=d["dialect"],
                architecture=arch,
                setting=choice["setting"],
                selected=best,
                grid=values,
            )
        )
    write_json(metadata_path, meta)
write_json(
    cfg.reports / "learning/thresholds.json",
    dict(
        protocol="Threshold grid chosen on the same development folds as model selection. Fixed-0.5 augmentation comparisons remain separate.",
        results=output,
    ),
)
