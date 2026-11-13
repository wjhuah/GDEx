"""Versioned recovery database and paired experiments against frozen validation rows."""

import sys, json, hashlib, sqlite3, copy
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config, write_json
from coptic_examples.corpus import Corpus
from coptic_examples.lexicon import load
from coptic_examples.citation_recovery import add_cdo_only_entries
from coptic_examples.inclusion import EntryFeatures, COLUMNS_INCLUSION
from coptic_examples.learning import (
    fit_inclusion,
    fit_score,
    predict,
    sample_weights,
    augment,
    crum_grade,
)
from coptic_examples.forest import fit as fit_forest
from coptic_examples.dataset import text_key
from coptic_examples.integrity import alignment_issue
from coptic_examples.ablation import expected_top_metric
from coptic_examples.text import real_translation, editorial_heading

cfg = Config()
out = cfg.reports / "本地复盘/新增数据模型比较"
out.mkdir(parents=True, exist_ok=True)
source = cfg.reports / "本地复盘/仅S异写兼容"
lex = load(cfg.lexicon)
add_cdo_only_entries(lex, json.loads((cfg.work / "cache/cdo_index.json").read_text()))
base = []
for dialect in ["sahidic", "bohairic"]:
    rows = list(map(json.loads, (cfg.derived / f"model_rows_{dialect}.jsonl").open()))
    manifest = json.loads((cfg.derived / f"fold_manifest_{dialect}.json").read_text())
    for r, m in zip(rows, manifest):
        assert (r["tla"], r["text_key"], r["candidate"]["id"]) == (
            m["tla"],
            m["text_key"],
            m["sentence_id"],
        )
        r["fold"] = m["fold"]
        r["x"] = [float(v) if v is not None else float("nan") for v in r["x"]]
    base += rows
family_folds = {r["family"]: r["fold"] for r in base if r["dialect"] == "sahidic"}
assert all(
    family_folds[r["family"]] == r["fold"] for r in base if r["dialect"] == "sahidic"
)
positions = {
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
extra_path = out / "新增训练记录.jsonl"
excluded = Counter()
extra = []
if extra_path.exists():
    extra = list(map(json.loads, extra_path.open()))
else:
    ef = EntryFeatures(cfg, lex, "sahidic")
    seen = {(r["dialect"], r["tla"], r["text_key"]) for r in base}
    with Corpus(cfg).connect() as db:
        for p in map(json.loads, (source / "恢复配对_待验证.jsonl").open()):
            c = json.loads(
                db.execute(
                    "select payload from sentence where id=?", (p["sentence_id"],)
                ).fetchone()[0]
            )
            e = lex[p["tla"]]
            if (
                alignment_issue(c)
                or not real_translation(c["translation"])
                or editorial_heading(c["translation"])
            ):
                excluded["文本或译文检查"] += 1
                continue
            key = ("sahidic", e.tla, text_key(c))
            if key in seen:
                excluded["已有同词条原文"] += 1
                continue
            seen.add(key)
            c.setdefault("target_alignment", {})[e.tla] = list(
                dict.fromkeys(t for span in p["spans"] for t in span)
            )
            r = dict(
                tla=e.tla,
                family=e.lemma,
                dialect="sahidic",
                text_key=key[2],
                candidate=c,
                y=1,
                ranks=[],
                sources=["crum_automatic"],
                expert_verified=False,
                citation_ids=p["citation_ids"],
            )
            r["x"] = [
                float(v) if v is not None else float("nan") for v in ef.vector(c, e)
            ]
            r["fold"] = family_folds.get(
                e.lemma, int(hashlib.sha256(e.lemma.encode()).hexdigest()[:8], 16) % 5
            )
            extra.append(r)
            if len(extra) % 500 == 0:
                print("features", len(extra), flush=True)
    extra_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in extra)
    )
# Recover ordering only where explicit source positions exist. Recalculate the combined Crum sequence.
combined = copy.deepcopy(base + extra)
groups = defaultdict(list)
for r in combined:
    if r["y"] > 0 and ("crum" in r["sources"] or "crum_automatic" in r["sources"]):
        groups[(r["dialect"], r["tla"])].append(r)
for group in groups.values():

    def position(r):
        ids = r.get("citation_ids", []) + [
            v.get("occurrence_id") for v in r.get("crum_refs", [])
        ]
        return min(
            (positions[i] for i in ids if i in positions), default=(99999, False, "", 0)
        )

    group.sort(key=position)
    for i, r in enumerate(group, 1):
        r["ranks"] = [v for v in r["ranks"] if v["source"] != "crum"] + [
            dict(
                source="crum",
                grade=crum_grade(i, len(group)),
                position=i,
                total=len(group),
            )
        ]
combined_extra = combined[len(base) :]
# Independent SQLite dataset: source provenance and original TT/translation/features stay available per row.
db = sqlite3.connect(out / "实验数据.sqlite3")
db.executescript(
    "DROP TABLE IF EXISTS example; DROP TABLE IF EXISTS metadata; CREATE TABLE example(id INTEGER PRIMARY KEY, cohort TEXT,dialect TEXT,tla TEXT,text_key TEXT,fold INTEGER,label INTEGER,payload TEXT,UNIQUE(cohort,dialect,tla,text_key)); CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);"
)
for cohort, rr in [("original", base), ("recovered", combined_extra)]:
    db.executemany(
        "insert into example(cohort,dialect,tla,text_key,fold,label,payload) values(?,?,?,?,?,?,?)",
        [
            (
                cohort,
                r["dialect"],
                r["tla"],
                r["text_key"],
                r["fold"],
                r["y"],
                json.dumps(r, ensure_ascii=False),
            )
            for r in rr
        ],
    )
db.execute(
    "insert into metadata values(?,?)", ("feature_names", json.dumps(COLUMNS_INCLUSION))
)
db.commit()
assert db.execute("pragma integrity_check").fetchone()[0] == "ok"
db.close()
write_json(
    out / "数据规模.json",
    dict(
        original={
            d: dict(
                Counter(
                    "positive" if r["y"] > 0 else "negative"
                    for r in base
                    if r["dialect"] == d
                )
            )
            for d in ["sahidic", "bohairic"]
        },
        recovered=len(extra),
        excluded=excluded,
        features=len(COLUMNS_INCLUSION),
        automatic_expert_verified=False,
    ),
)


def ranking(rows, scores):
    g = defaultdict(list)
    for r, s in zip(rows, scores):
        if r["y"] > 0:
            for v in r["ranks"]:
                g[(r["tla"], v["source"])].append((v["grade"], float(s)))
    result = {}
    for source_name in ["tab", "crum"]:
        vals = [
            expected_top_metric([a for a, b in v], [b for a, b in v], 6)
            for (t, src), v in g.items()
            if src == source_name and len(set(a for a, b in v)) > 1
        ]
        pairs = [
            (a[0] - b[0], a[1] - b[1])
            for (t, src), v in g.items()
            if src == source_name
            for i, a in enumerate(v)
            for b in v[i + 1 :]
            if a[0] != b[0]
        ]
        result[source_name] = dict(
            groups=len(vals),
            ndcg6=float(np.mean(vals)) if vals else None,
            pair_accuracy=(
                float(
                    np.mean(
                        [1 if a * b > 0 else 0.5 if b == 0 else 0 for a, b in pairs]
                    )
                )
                if pairs
                else None
            ),
        )
    return result


results = []
importance = []
for dialect in ["sahidic", "bohairic"]:
    data = [r for r in base if r["dialect"] == dialect]
    updated = [r for r in combined if r["dialect"] == dialect]
    ef = EntryFeatures(cfg, lex, dialect)
    memo = {}
    settings = (
        [("original", False), ("recovered", False), ("recovered_augmented", True)]
        if dialect == "sahidic"
        else [("original", False)]
    )
    for version, augmented in settings:
        for fold in range(5):
            val = [r for r in data if r["fold"] == fold]
            held = {r["text_key"] for r in val}
            if len(set(r["y"] > 0 for r in val)) < 2:
                continue
            tr = [
                r
                for r in (data if version == "original" else updated)
                if r["fold"] != fold and r["text_key"] not in held
            ]
            extra_neg = augment(tr, 0.5, "mixed", ef, lex, memo) if augmented else []
            extra_neg = [
                r
                for r in extra_neg
                if r["text_key"] not in held
                and r["family"] not in {v["family"] for v in val}
            ]
            train = tr + extra_neg
            x = np.array([r["x"] for r in train], dtype=float)
            y = np.array([r["y"] > 0 for r in train])
            vx = np.array([r["x"] for r in val], dtype=float)
            vy = np.array([r["y"] > 0 for r in val])
            ranker = fit_score(tr, False)
            rs = predict(ranker, val)
            joint = fit_score(train, True)
            methods = (
                ["RandomForest", "LightGBM"]
                if augmented
                else ["DecisionTree", "RandomForest", "LightGBM", "HistGBDT"]
            )
            for method in methods:
                if method == "LightGBM":
                    model = fit_inclusion(train)
                    proba = predict(model, val)
                    imp = model.feature_importance("gain")
                elif method == "RandomForest":
                    model = fit_forest(train)
                    proba = model.predict_proba(vx)[:, 1]
                    imp = model.named_steps["forest"].feature_importances_
                elif method == "DecisionTree":
                    model = make_pipeline(
                        SimpleImputer(keep_empty_features=True),
                        DecisionTreeClassifier(
                            max_depth=16, min_samples_leaf=5, random_state=42
                        ),
                    )
                    model.fit(
                        x,
                        y,
                        decisiontreeclassifier__sample_weight=sample_weights(train),
                    )
                    proba = model.predict_proba(vx)[:, 1]
                    imp = model[-1].feature_importances_
                else:
                    model = HistGradientBoostingClassifier(
                        max_iter=120,
                        max_leaf_nodes=15,
                        l2_regularization=2,
                        random_state=42,
                    )
                    model.fit(x, y, sample_weight=sample_weights(train))
                    proba = model.predict_proba(vx)[:, 1]
                    imp = None
                cls = dict(
                    auc=float(roc_auc_score(vy, proba)),
                    average_precision=float(average_precision_score(vy, proba)),
                    macro_f1=float(f1_score(vy, proba > 0.5, average="macro")),
                    precision=float(precision_score(vy, proba > 0.5, zero_division=0)),
                    recall=float(recall_score(vy, proba > 0.5, zero_division=0)),
                )
                # Inclusion first, learned preference second; identical test candidates throughout.
                order = np.where(proba > 0.5, rs, rs - 1000000)
                item = dict(
                    dialect=dialect,
                    version=version,
                    fold=fold,
                    method=method,
                    train=len(tr),
                    augmented=len(extra_neg),
                    validation=len(val),
                    classification=cls,
                    ranking_only=ranking(val, rs),
                    two_stage=ranking(val, order),
                    unified=ranking(val, predict(joint, val)),
                )
                results.append(item)
                if imp is not None:
                    imp = np.asarray(imp, dtype=float)
                    imp = imp / max(imp.sum(), 1e-12)
                    importance.append(
                        dict(
                            dialect=dialect,
                            version=version,
                            fold=fold,
                            method=method,
                            importance=dict(zip(COLUMNS_INCLUSION, map(float, imp))),
                        )
                    )
                write_json(out / "逐折结果.json", results)
                write_json(out / "特征重要性.json", importance)
                print(dialect, version, fold, method, cls, flush=True)
write_json(out / "完成标记.json", dict(completed=True, fold_results=len(results)))
