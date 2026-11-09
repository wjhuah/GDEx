#!/usr/bin/env python3
"""Paired inclusion comparison on frozen spelling-family folds."""

import sys, json, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import joblib
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config, write_json
from coptic_examples.learning import fit_inclusion, fit_score, predict, augment
from coptic_examples.forest import fit
from coptic_examples.lexicon import load
from coptic_examples.inclusion import EntryFeatures, COLUMNS_INCLUSION
from coptic_examples.ablation import expected_top_metric


def metrics(y, p, threshold):
    z = np.asarray(p) > threshold
    return dict(
        auc=float(roc_auc_score(y, p)),
        average_precision=float(average_precision_score(y, p)),
        macro_f1=float(f1_score(y, z, average="macro", zero_division=0)),
        precision=float(precision_score(y, z, zero_division=0)),
        recall=float(recall_score(y, z, zero_division=0)),
        threshold=float(threshold),
    )


def ranking(rows, scores, inclusion=None, threshold=0.5):
    groups = defaultdict(list)
    for i, (r, s) in enumerate(zip(rows, scores)):
        for rank in r["ranks"]:
            if r["y"] > 0:
                groups[(r["tla"], rank["source"])].append(
                    (
                        rank["grade"],
                        s if inclusion is None or inclusion[i] > threshold else -1e6,
                    )
                )
    return {
        source: dict(groups=len(v), ndcg_at_6=float(np.mean(v)) if v else None)
        for source in ("tab", "crum")
        for v in [
            [
                expected_top_metric([a for a, b in g], [b for a, b in g], 6)
                for (t, s), g in groups.items()
                if s == source and len({a for a, b in g}) > 1
            ]
        ]
    }


def main():
    cfg = Config()
    lex = load(cfg.lexicon)
    report = []
    for dialect, ratio in [("sahidic", 0.5), ("bohairic", 0.25)]:
        data = [
            json.loads(s) for s in (cfg.derived / f"model_rows_{dialect}.jsonl").open()
        ]
        for r in data:
            r["x"] = [float(v) if v is not None else float("nan") for v in r["x"]]
        manifest = json.loads(
            (cfg.derived / f"fold_manifest_{dialect}.json").read_text()
        )
        assert len(data) == len(manifest)
        for r, m in zip(data, manifest):
            assert (r["tla"], r["text_key"], r["candidate"]["id"]) == (
                m["tla"],
                m["text_key"],
                m["sentence_id"],
            )
            r["fold"] = m["fold"]
        ef = EntryFeatures(cfg, lex, dialect)
        memo = {}
        oof = []
        counts = []
        for fold in range(5):
            val = [r for r in data if r["fold"] == fold]
            held = {r["text_key"] for r in val}
            tr = [r for r in data if r["fold"] != fold and r["text_key"] not in held]
            extra = augment(tr, ratio, "mixed", ef, lex, memo)
            assert not any(r["text_key"] in held for r in extra)
            old = fit_inclusion(tr + extra)
            new = fit(tr + extra)
            ranker = fit_score(tr, False)
            x = np.asarray([r["x"] for r in val])
            oldp = predict(old, val)
            newp = new.predict_proba(x)[:, 1]
            rs = predict(ranker, val)
            oof.extend(
                dict(
                    tla=r["tla"],
                    text_key=r["text_key"],
                    id=r["candidate"]["id"],
                    y=r["y"],
                    ranks=r["ranks"],
                    fold=fold,
                    lgb=float(a),
                    rf=float(b),
                    rank=float(s),
                )
                for r, a, b, s in zip(val, oldp, newp, rs)
            )
            counts.append(
                dict(
                    fold=fold, train=len(tr), augmented=len(extra), validation=len(val)
                )
            )
            print(dialect, fold, "done", flush=True)
        y = np.asarray([r["y"] > 0 for r in oof])
        summary = {}
        for name in ("lgb", "rf"):
            p = np.asarray([r[name] for r in oof])
            grid = [metrics(y, p, t) for t in np.arange(0.05, 0.996, 0.01)]
            best = max(grid, key=lambda r: r["macro_f1"])
            summary[name] = dict(
                fixed_05=metrics(y, p, 0.5), development_threshold=best
            )
        final_extra = augment(data, ratio, "mixed", ef, lex, memo)
        model = fit(data + final_extra)
        folder = cfg.work / "models" / dialect
        joblib.dump(model, folder / "inclusion_rf.joblib", compress=3)
        meta = json.loads((folder / "learning_v2.json").read_text())
        meta["inclusion_algorithm"] = "random_forest"
        meta["thresholds"]["two_models"] = summary["rf"]["development_threshold"][
            "threshold"
        ]
        write_json(folder / "learning_v2.json", meta)
        importance = sorted(
            [
                dict(feature=f, mdi=float(v))
                for f, v in zip(
                    COLUMNS_INCLUSION, model.named_steps["forest"].feature_importances_
                )
            ],
            key=lambda r: -r["mdi"],
        )
        rankimportance = {}
        import lightgbm as lgb

        for name in ("ranking_v2", "joint_v2", "inclusion_v2"):
            booster = lgb.Booster(model_file=str(folder / (name + ".txt")))
            g = booster.feature_importance("gain")
            sp = booster.feature_importance("split")
            rankimportance[name] = sorted(
                [
                    dict(
                        feature=f,
                        gain_share=float(v / sum(g)) if sum(g) else 0,
                        split=int(n),
                    )
                    for f, v, n in zip(booster.feature_name(), g, sp)
                ],
                key=lambda r: -r["gain_share"],
            )
        result = dict(
            dialect=dialect,
            rows=len(data),
            positive=int(sum(y)),
            folds=counts,
            final_augmentation=len(final_extra),
            classification=summary,
            ranking_positive_candidates=ranking(oof, [r["rank"] for r in oof]),
            forest_importance=importance,
            lightgbm_importance=rankimportance,
        )
        report.append(result)
        write_json(cfg.reports / f"forest/oof_{dialect}.json", oof)
        write_json(
            cfg.reports / "forest/comparison.json",
            dict(
                protocol="Fixed original candidate rows; identical five spelling-family folds; duplicate texts purged from training; same train-only mixed augmentation; thresholds selected on development OOF. Ranking model unchanged. Token sense head and UI filters excluded from this classifier comparison.",
                results=report,
            ),
        )
        ef.close()
    lines = [
        "# Random forest comparison",
        "",
        "Development cross-validation; thresholds were selected on these folds.",
        "",
        "| Dialect | Model | AUC | AP | Macro F1 @0.5 | Precision @0.5 | Recall @0.5 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in report:
        for n, s in r["classification"].items():
            m = s["fixed_05"]
            lines.append(
                "| "
                + r["dialect"]
                + " | "
                + n
                + " | "
                + " | ".join(
                    f"{m[k]:.4f}"
                    for k in [
                        "auc",
                        "average_precision",
                        "macro_f1",
                        "precision",
                        "recall",
                    ]
                )
                + " |"
            )
        lines += [
            "",
            f"{r['dialect']} positive-only preference ranking (same model for both classifiers): `{r['ranking_positive_candidates']}`.",
            "",
            "## " + r["dialect"] + " feature importance",
            "",
            "| Feature | Forest MDI | Ranking gain | Unified gain |",
            "| --- | --- | --- | --- |",
        ]
        ranks = {
            v["feature"]: v["gain_share"]
            for v in r["lightgbm_importance"]["ranking_v2"]
        }
        joint = {
            v["feature"]: v["gain_share"] for v in r["lightgbm_importance"]["joint_v2"]
        }
        lines += [
            f"| {v['feature']} | {v['mdi']:.2%} | {ranks.get(v['feature'],0):.2%} | {joint.get(v['feature'],0):.2%} |"
            for v in r["forest_importance"]
        ]
    lines += [
        "",
        "MDI and gain are training-based importance measures on different scales. Correlated features share importance. Classification results cover original labeled candidates, before token-sense and display policies. Ranking NDCG uses positive ordered source examples; the classifier replacement leaves this preference model unchanged.",
    ]
    (cfg.reports / "forest/REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
