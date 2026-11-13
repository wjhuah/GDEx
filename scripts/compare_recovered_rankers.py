"""Preference algorithms on the same frozen candidates and positive training rows."""

import sys, json, sqlite3, copy
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config, write_json
from coptic_examples.learning import fit_score, predict, rank_observations, crum_grade
from coptic_examples.inclusion import COLUMNS_INCLUSION
from coptic_examples.ablation import expected_top_metric

cfg = Config()
out = cfg.reports / "本地复盘/新增数据模型比较"
db = sqlite3.connect(out / "实验数据.sqlite3")
base = [
    json.loads(v[0])
    for v in db.execute("select payload from example where cohort='original'")
]
extra = [
    json.loads(v[0])
    for v in db.execute("select payload from example where cohort='recovered'")
]
db.close()
combined = copy.deepcopy(base + extra)
pos = {
    r["id"]: (
        int("".join(filter(str.isdigit, r["page"])) or 0),
        r["page"].endswith("b"),
        r["file"],
        r["char_start"],
    )
    for r in map(
        json.loads, (cfg.derived / "complete_citations/citations.jsonl").open()
    )
}
groups = defaultdict(list)
for r in combined:
    if r["y"] > 0 and ("crum" in r["sources"] or "crum_automatic" in r["sources"]):
        groups[(r["dialect"], r["tla"])].append(r)
for rr in groups.values():

    def key(r):
        return min(
            (
                pos[i]
                for i in r.get("citation_ids", [])
                + [v.get("occurrence_id") for v in r.get("crum_refs", [])]
                if i in pos
            ),
            default=(99999, False, "", 0),
        )

    rr.sort(key=key)
    for i, r in enumerate(rr, 1):
        r["ranks"] = [p for p in r["ranks"] if p["source"] != "crum"] + [
            dict(source="crum", grade=crum_grade(i, len(rr)))
        ]
results = []
importance = []
for dialect in ["sahidic", "bohairic"]:
    data = [r for r in base if r["dialect"] == dialect]
    for version in (
        ["original", "recovered"] if dialect == "sahidic" else ["original"]
    ):
        for fold in range(5):
            val = [r for r in data if r["fold"] == fold and r["y"] > 0]
            held = {r["text_key"] for r in data if r["fold"] == fold}
            tr = [
                r
                for r in (base if version == "original" else combined)
                if r["dialect"] == dialect
                and r["fold"] != fold
                and r["text_key"] not in held
            ]
            obs = [r for r in rank_observations(tr) if r["y"] > 0 and not r["bounded"]]
            if not obs or not val:
                continue
            x = np.array([r["x"] for r in obs], dtype=float)
            y = np.array([r["target"] for r in obs])
            vx = np.array([r["x"] for r in val], dtype=float)
            for method in ["LightGBM", "RandomForest", "HistGBDT"]:
                if method == "LightGBM":
                    m = fit_score(tr, False)
                    scores = predict(m, val)
                    imp = m.feature_importance("gain")
                elif method == "RandomForest":
                    m = make_pipeline(
                        SimpleImputer(keep_empty_features=True),
                        RandomForestRegressor(
                            n_estimators=100,
                            max_depth=16,
                            min_samples_leaf=5,
                            n_jobs=4,
                            random_state=42,
                        ),
                    )
                    m.fit(x, y)
                    scores = m.predict(vx)
                    imp = m[-1].feature_importances_
                else:
                    m = HistGradientBoostingRegressor(
                        max_iter=120,
                        max_leaf_nodes=15,
                        l2_regularization=2,
                        random_state=42,
                    )
                    m.fit(x, y)
                    scores = m.predict(vx)
                    imp = None
                g = defaultdict(list)
                for r, s in zip(val, scores):
                    for p in r["ranks"]:
                        g[(r["tla"], p["source"])].append((p["grade"], float(s)))
                metrics = {}
                for source in ["tab", "crum"]:
                    values = [
                        v
                        for (t, src), v in g.items()
                        if src == source and len(set(a for a, b in v)) > 1
                    ]
                    ndcg = [
                        expected_top_metric([a for a, b in v], [b for a, b in v], 6)
                        for v in values
                    ]
                    pairs = [
                        (a[0] - b[0], a[1] - b[1])
                        for v in values
                        for i, a in enumerate(v)
                        for b in v[i + 1 :]
                        if a[0] != b[0]
                    ]
                    metrics[source] = dict(
                        groups=len(values),
                        ndcg6=float(np.mean(ndcg)) if ndcg else None,
                        pair_accuracy=(
                            float(
                                np.mean(
                                    [
                                        1 if a * b > 0 else 0.5 if b == 0 else 0
                                        for a, b in pairs
                                    ]
                                )
                            )
                            if pairs
                            else None
                        ),
                    )
                results.append(
                    dict(
                        dialect=dialect,
                        version=version,
                        fold=fold,
                        method=method,
                        training_rank_observations=len(obs),
                        ranking=metrics,
                    )
                )
                if imp is not None:
                    imp = imp / max(sum(imp), 1e-12)
                    importance.append(
                        dict(
                            dialect=dialect,
                            version=version,
                            fold=fold,
                            method=method,
                            importance=dict(zip(COLUMNS_INCLUSION, map(float, imp))),
                        )
                    )
                write_json(out / "独立排序逐折结果.json", results)
                write_json(out / "排序特征重要性.json", importance)
                print(dialect, version, fold, method, metrics, flush=True)
write_json(out / "排序完成标记.json", dict(completed=True, fold_results=len(results)))
