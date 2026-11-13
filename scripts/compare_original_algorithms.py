"""Frozen original-data classification and preference comparisons; no deployment writes."""

import os, sys, json, hashlib, gzip, time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.inclusion import COLUMNS_INCLUSION
from coptic_examples.ablation import expected_top_metric

root = Path(__file__).resolve().parents[1]
out = root / "reports/本地复盘/原数据八算法对照"
out.mkdir(parents=True, exist_ok=True)


def save(name, value):
    (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def factory(name, classification):
    common = dict(random_state=42)
    if name == "Linear":
        return make_pipeline(
            SimpleImputer(keep_empty_features=True),
            StandardScaler(),
            (
                LogisticRegression(max_iter=1000, **common)
                if classification
                else Ridge(alpha=2)
            ),
        )
    if name == "DecisionTree":
        m = (DecisionTreeClassifier if classification else DecisionTreeRegressor)(
            max_depth=16, min_samples_leaf=5, **common
        )
    elif name in ["RandomForest", "ExtraTrees"]:
        cls = (
            {
                "RandomForest": RandomForestClassifier,
                "ExtraTrees": ExtraTreesClassifier,
            }[name]
            if classification
            else {
                "RandomForest": RandomForestRegressor,
                "ExtraTrees": ExtraTreesRegressor,
            }[name]
        )
        m = cls(
            n_estimators=200 if classification else 100,
            max_depth=16,
            min_samples_leaf=5,
            max_features="sqrt" if classification else 1.0,
            n_jobs=4,
            **common,
        )
    elif name == "HistGBDT":
        m = (
            HistGradientBoostingClassifier
            if classification
            else HistGradientBoostingRegressor
        )(max_iter=120, max_leaf_nodes=15, l2_regularization=2, **common)
    elif name == "LightGBM":
        m = (LGBMClassifier if classification else LGBMRegressor)(
            n_estimators=120,
            num_leaves=15,
            min_child_samples=20,
            learning_rate=0.05,
            reg_lambda=2,
            n_jobs=4,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
            **common,
        )
    elif name == "XGBoost":
        m = (XGBClassifier if classification else XGBRegressor)(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            reg_lambda=2,
            tree_method="hist",
            n_jobs=4,
            **common,
        )
    elif name == "CatBoost":
        m = (CatBoostClassifier if classification else CatBoostRegressor)(
            iterations=120,
            depth=4,
            learning_rate=0.05,
            l2_leaf_reg=2,
            random_seed=42,
            thread_count=4,
            verbose=False,
            allow_writing_files=False,
        )
    return make_pipeline(SimpleImputer(keep_empty_features=True), m)


def fit(model, x, y, weights):
    key = model.steps[-1][0] + "__sample_weight"
    model.fit(x, y, **{key: weights})
    return model


def rank_metrics(rows, scores):
    groups = defaultdict(list)
    for r, score in zip(rows, scores):
        if r["y"] > 0:
            for v in r["ranks"]:
                groups[r["tla"], v["source"]].append((v["grade"], float(score)))
    result = {}
    for source in ["tab", "crum"]:
        gg = [
            g
            for (t, s), g in groups.items()
            if s == source and len(set(a for a, b in g)) > 1
        ]
        ndcg = [
            expected_top_metric([a for a, b in g], [b for a, b in g], 6) for g in gg
        ]
        wins = []
        for g in gg:
            for i, (a, b) in enumerate(g):
                for c, d in g[i + 1 :]:
                    if a != c:
                        wins.append(
                            1 if (a - c) * (b - d) > 0 else 0.5 if b == d else 0
                        )
        result[source] = dict(
            groups=len(gg),
            ndcg6=float(np.mean(ndcg)) if ndcg else None,
            pair_accuracy=float(np.mean(wins)) if wins else None,
        )
    return result


results = []
importance = []
provenance = {}
methods = [
    "Linear",
    "DecisionTree",
    "RandomForest",
    "ExtraTrees",
    "HistGBDT",
    "LightGBM",
    "XGBoost",
    "CatBoost",
]
with gzip.open(out / "留出预测.jsonl.gz", "wt") as predictions:
    for dialect in ["sahidic", "bohairic"]:
        file = root / f"data/derived/model_rows_{dialect}.jsonl"
        mf = root / f"data/derived/fold_manifest_{dialect}.json"
        provenance[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
        provenance[str(mf)] = hashlib.sha256(mf.read_bytes()).hexdigest()
        manifest = json.loads(mf.read_text())
        data = []
        for line, m in zip(file.open(), manifest):
            r = json.loads(line)
            assert (r["tla"], r["text_key"], r["candidate"]["id"]) == (
                m["tla"],
                m["text_key"],
                m["sentence_id"],
            )
            data.append(
                dict(
                    tla=r["tla"],
                    family=r["family"],
                    text_key=r["text_key"],
                    id=m["sentence_id"],
                    x=r["x"],
                    y=r["y"],
                    ranks=r["ranks"],
                    fold=m["fold"],
                )
            )
        assert len(data) == len(manifest)
        matrix = np.array(
            [[float(v) if v is not None else np.nan for v in r["x"]] for r in data]
        )
        labels = np.array([r["y"] > 0 for r in data])
        assert matrix.shape[1] == len(COLUMNS_INCLUSION)
        for fold in range(5):
            vi = np.array([i for i, r in enumerate(data) if r["fold"] == fold])
            held = {data[i]["text_key"] for i in vi}
            ti = np.array(
                [
                    i
                    for i, r in enumerate(data)
                    if r["fold"] != fold and r["text_key"] not in held
                ]
            )
            tr = [data[i] for i in ti]
            val = [data[i] for i in vi]
            assert not set(r["family"] for r in tr) & set(r["family"] for r in val)
            y = labels[ti]
            counts = np.bincount(y.astype(int), minlength=2)
            weights = np.array([len(y) / (2 * counts[int(v)]) for v in y])
            rx = []
            ry = []
            for idx in ti:
                r = data[idx]
                if r["y"] > 0:
                    for p in r["ranks"]:
                        rx.append(matrix[idx])
                        ry.append(p["grade"])
            for method in methods:
                start = time.time()
                classifier = fit(factory(method, True), matrix[ti], y, weights)
                prob = classifier.predict_proba(matrix[vi])[:, 1]
                z = prob > 0.5
                vy = labels[vi]
                cls = dict(
                    auc=float(roc_auc_score(vy, prob)) if len(set(vy)) > 1 else None,
                    average_precision=(
                        float(average_precision_score(vy, prob)) if any(vy) else None
                    ),
                    macro_f1=float(f1_score(vy, z, average="macro", zero_division=0)),
                    precision=float(precision_score(vy, z, zero_division=0)),
                    recall=float(recall_score(vy, z, zero_division=0)),
                )
                ranker = fit(
                    factory(method, False),
                    np.asarray(rx),
                    np.asarray(ry),
                    np.ones(len(ry)),
                )
                score = ranker.predict(matrix[vi])
                gated = np.where(prob > 0.5, score, score - 1e6)
                item = dict(
                    dialect=dialect,
                    fold=fold,
                    method=method,
                    train=len(ti),
                    validation=len(vi),
                    train_positive=int(sum(y)),
                    validation_positive=int(sum(vy)),
                    classification=cls,
                    ranking=rank_metrics(val, score),
                    two_stage=rank_metrics(val, gated),
                    seconds=time.time() - start,
                )
                results.append(item)
                for role, model in [
                    ("classification", classifier),
                    ("ranking", ranker),
                ]:
                    m = model[-1]
                    imp = getattr(m, "feature_importances_", None)
                    if imp is None and hasattr(m, "coef_"):
                        imp = np.abs(m.coef_).reshape(-1)
                    if imp is not None:
                        imp = np.array(imp, dtype=float)
                        imp = imp / max(imp.sum(), 1e-12)
                        importance.append(
                            dict(
                                dialect=dialect,
                                fold=fold,
                                method=method,
                                role=role,
                                values=dict(zip(COLUMNS_INCLUSION, map(float, imp))),
                            )
                        )
                for r, pred, rs in zip(val, prob, score):
                    predictions.write(
                        json.dumps(
                            dict(
                                dialect=dialect,
                                fold=fold,
                                method=method,
                                tla=r["tla"],
                                text_key=r["text_key"],
                                id=r["id"],
                                label=r["y"],
                                probability=float(pred),
                                rank_score=float(rs),
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                predictions.flush()
                save("逐折指标.json", results)
                save("逐折特征重要性.json", importance)
                print(dialect, fold, method, cls, flush=True)
save(
    "实验协议.json",
    dict(
        input_sha256=provenance,
        features=COLUMNS_INCLUSION,
        classification_threshold=0.5,
        methods=methods,
        augmentation=False,
        new_recovered_data=False,
        seed=42,
        split="frozen spelling-family folds; repeated validation texts removed from training",
        hyperparameter_search=False,
    ),
)
save("完成.json", dict(completed=True, result_rows=len(results)))
