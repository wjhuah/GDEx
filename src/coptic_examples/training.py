"""Reproducible lemma-disjoint, sentence-purged ranking experiments."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb
from .config import write_json, database
from .corpus import Corpus
from .dataset import read_rows
from .embeddings import Embeddings, MODEL
from .features import Features, GDEX, UD, SM
from .text import lemma_key, real_translation, sm_freq, norm


def ndcg(grades, scores, k=6):
    grades = np.asarray(grades, dtype=float)
    scores = np.asarray(scores)
    size = min(k, len(grades))
    ideal = np.sort(grades)[::-1][:size]
    discount = np.log2(np.arange(2, size + 2))
    denom = float(np.sum(ideal / discount))
    return (
        float(
            np.sum(grades[np.argsort(-scores, kind="stable")][:size] / discount) / denom
        )
        if denom
        else None
    )


def pair_accuracy(grades, scores):
    wins = []
    for i in range(len(grades)):
        for j in range(i):
            if grades[i] != grades[j]:
                wins.append(
                    1.0
                    if (grades[i] - grades[j]) * (scores[i] - scores[j]) > 0
                    else 0.5 if scores[i] == scores[j] else 0.0
                )
    return float(np.mean(wins)) if wins else None


def feature_data(config):
    rows = prepare_rows(read_rows(config))
    rows = [r for r in rows if r["grade"] is not None and r["gloss"]]
    fe = Features(Corpus(config).frequencies())
    embed = Embeddings(config.vectors)
    # Reuse actual vectors only (never legacy cosine/labels); hashes use the same NFKC whitespace norm.
    legacy = config.source / "crum3/deploy/crum4/cache/embeddings.json"
    if legacy.exists():
        import sqlite3

        old = json.loads(legacy.read_text())
        if old.get("model") == MODEL:
            n = 0
            with database(config.vectors) as db:
                for kind, role in [
                    ("definitions", "definition"),
                    ("sentences", "sentence"),
                ]:
                    for key, v in old.get("embeddings", {}).get(kind, {}).items():
                        if len(v) != 1024 or not np.isfinite(v).all():
                            continue
                        db.execute(
                            "INSERT OR IGNORE INTO vector VALUES(?,?,?,?)",
                            (MODEL, role, key, json.dumps(v)),
                        )
                        n += 1
            write_json(
                config.reports / "vector_cache_provenance.json",
                dict(
                    source=str(legacy),
                    sha256=hashlib.sha256(legacy.read_bytes()).hexdigest(),
                    valid_vectors=n,
                    protocol="NFKC/whitespace normalization; no query prefix; normalized embeddings; model namespace",
                ),
            )
    glosses = sorted(set(r["gloss"] for r in rows))
    translations = sorted(
        set(
            norm(r["candidate"]["translation"])
            for r in rows
            if real_translation(r["candidate"]["translation"])
        )
    )
    print(
        f"Computing features: {len(rows)} rows, {len(glosses)} unique definitions, {len(translations)} unique translations",
        flush=True,
    )
    gv = dict(zip(glosses, embed.encode(glosses, "definition"), strict=True))
    tv = dict(zip(translations, embed.encode(translations), strict=True))
    result = []
    for i, r in enumerate(rows):
        c = r["candidate"]
        f = fe.extract(c, r["lemma"])
        tr = norm(c["translation"])
        f["sm_freq"] = sm_freq(r["gloss"], tr) if real_translation(tr) else 0.0
        f["sm_cosine"] = float(gv[r["gloss"]] @ tv[tr]) if tr in tv else 0.0
        f.update(
            row_id=i,
            TLA=r["TLA"],
            lemma=r["lemma"],
            group=r["group"],
            evidence=r["evidence"],
            text_key=r["text_key"],
            grade=r["grade"],
            weight=r["weight"],
        )
        result.append(f)
    pd.DataFrame(result).to_csv(config.derived / "features.tsv", sep="\t", index=False)
    # Corresponding metadata remains in a stable row_id file, independent of future source changes.
    with (config.derived / "feature_rows.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return dict(
        rows=len(rows),
        definitions=len(glosses),
        translations=len(translations),
        features=len(GDEX + UD + SM),
    )


def prepare_rows(evidence_rows):
    """Inclusion and ordinal preference are different expert tasks.

    Unranked expert selections are known accepted examples, although their order
    is unknown. Give them binary acceptance evidence in a SEPARATE query.
    """
    rows = []
    for r in evidence_rows:
        if r["evidence"] == "manual":
            if r["grade"] is not None:
                rows.append(r)
            suffix = "|" + r["dialect_id"] if r.get("dialect_id") else ""
            rows.append(
                dict(
                    r,
                    group=r["TLA"] + "|manual_acceptance" + suffix,
                    grade=10,
                    evidence="manual_acceptance",
                    signed_label=None,
                    acceptance=1,
                )
            )
        elif r["evidence"] == "manual_forbidden_rule":
            suffix = "|" + r["dialect_id"] if r.get("dialect_id") else ""
            rows.append(
                dict(r, group=r["TLA"] + "|manual_acceptance" + suffix, acceptance=0)
            )
        else:
            rows.append(r)
    return rows


def load_feature_rows(config):
    with (config.derived / "feature_rows.jsonl").open(encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def make_negatives(rows, features, mode, embed):
    if mode == "none":
        return [], []
    bylemma = defaultdict(list)
    bygroup = defaultdict(list)
    existing = defaultdict(set)
    for r, f in zip(rows, features, strict=True):
        existing[r["TLA"]].add(r["text_key"])
        if r["grade"] > 0 and real_translation(r["candidate"]["translation"]):
            bylemma[lemma_key(r["lemma"])].append((r, f))
            bygroup[r["group"]].append((r, f))
    pairs = {
        frozenset(("C992", "C998"))
    }  # only the expert-grounded disjoint hypothesis, no gloss-overlap shortcut
    generated = []
    numeric = []
    jobs = []
    for group, targets in sorted(bygroup.items()):
        target = targets[0][0]
        pool = bylemma[lemma_key(target["lemma"])]
        seen = set(existing[target["TLA"]])
        possible = []
        for source, source_f in pool:
            if source["TLA"] == target["TLA"] or source["text_key"] in seen:
                continue
            if (
                mode == "conservative"
                and frozenset((target["TLA"], source["TLA"])) not in pairs
            ):
                continue
            possible.append((source, source_f))
            seen.add(source["text_key"])
        possible.sort(
            key=lambda item: hashlib.sha256(
                (group + item[0]["text_key"]).encode()
            ).hexdigest()
        )
        for source, source_f in possible[: len(targets)]:
            r = dict(
                target,
                candidate=source["candidate"],
                text_key=source["text_key"],
                grade=0,
                signed_label=-1,
                evidence="inferred_cross_sense_" + mode,
                source_tla=source["TLA"],
                weight=0.35,
            )
            f = dict(
                source_f,
                TLA=target["TLA"],
                lemma=target["lemma"],
                group=group,
                grade=0,
                weight=0.35,
                evidence=r["evidence"],
            )
            f["sm_freq"] = sm_freq(target["gloss"], source["candidate"]["translation"])
            generated.append(r)
            numeric.append(f)
            jobs.append((target["gloss"], norm(source["candidate"]["translation"])))
    if jobs:
        q = embed.encode([x[0] for x in jobs], "definition")
        v = embed.encode([x[1] for x in jobs])
        for f, cosine in zip(numeric, np.sum(q * v, axis=1), strict=True):
            f["sm_cosine"] = float(cosine)
    return generated, numeric


def fit(frame, columns, rounds=200):
    frame = frame.sort_values(["group", "text_key"], kind="stable")
    sizes = frame.groupby("group", sort=False).size().tolist()
    dataset = lgb.Dataset(
        frame[columns].astype(float),
        label=frame["grade"].astype(int),
        weight=frame["weight"],
        group=sizes,
        feature_name=columns,
    )
    params = dict(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[6],
        label_gain=list(range(31)),
        num_leaves=15,
        min_data_in_leaf=20,
        learning_rate=0.05,
        num_threads=4,
        verbosity=-1,
        seed=42,
        deterministic=True,
        force_col_wise=True,
        lambdarank_truncation_level=10,
    )
    return lgb.train(params, dataset, num_boost_round=rounds)


def evaluate_frame(frame, scores):
    frame = frame.copy()
    frame["pred"] = scores
    out = []
    for group, g in frame.groupby("group", sort=True):
        if len(g) < 2 or g["grade"].nunique() < 2:
            continue
        out.append(
            dict(
                group=group,
                TLA=g["TLA"].iloc[0],
                evidence=g["evidence"].iloc[0],
                n=len(g),
                ndcg_at_6=ndcg(g["grade"], g["pred"]),
                pairwise_accuracy=pair_accuracy(
                    g["grade"].to_numpy(), g["pred"].to_numpy()
                ),
            )
        )
    return out


def experiment(config, folds=5, rounds=200):
    frame = pd.read_csv(
        config.derived / "features.tsv", sep="\t", keep_default_na=False
    )
    rows = load_feature_rows(config)

    # Group both TLA homographs and spelling variants before the split.
    def family(lemma):
        return __import__("re").sub(r"([ⲁⲉⲏⲓⲟⲱⲩ])\1", r"\1", lemma_key(lemma))

    families = sorted(set(frame["lemma"].map(family)))
    rng = np.random.default_rng(42)
    rng.shuffle(families)
    foldmap = {f: i % folds for i, f in enumerate(families)}
    frame["fold"] = [foldmap[family(x)] for x in frame["lemma"]]
    setups = [
        ("gdex", GDEX, "none"),
        ("gdex_ud", GDEX + UD, "none"),
        ("gdex_ud_sm", GDEX + UD + SM, "none"),
        ("naive_homograph_negatives", GDEX + UD + SM, "naive"),
        ("conservative_barley_father_negatives", GDEX + UD + SM, "conservative"),
    ]
    embed = Embeddings(config.vectors)
    allmetrics = []
    foldstats = []
    oof = []
    for fold in range(folds):
        val = frame[frame["fold"] == fold].copy()
        # Remove ANY training row whose Coptic span is also in held-out validation, even at another lemma.
        val_text = set(val["text_key"])
        mask = (frame["fold"] != fold) & ~frame["text_key"].isin(val_text)
        train = frame[mask].copy()
        indices = train["row_id"].astype(int).tolist()
        trainrows = [rows[i] for i in indices]
        trainfeatures = train.to_dict("records")
        assert not set(train["text_key"]) & val_text
        foldstats.append(
            dict(
                fold=fold,
                train_rows=len(train),
                validation_rows=len(val),
                purged_shared_sentence_rows=int(
                    ((frame["fold"] != fold) & frame["text_key"].isin(val_text)).sum()
                ),
            )
        )
        random_metrics = evaluate_frame(
            val, np.random.default_rng(1000 + fold).random(len(val))
        )
        for m in random_metrics:
            m.update(setup="random", fold=fold)
        allmetrics.extend(random_metrics)
        for name, columns, mode in setups:
            negrows, negfeatures = make_negatives(trainrows, trainfeatures, mode, embed)
            augmented = (
                pd.concat([train, pd.DataFrame(negfeatures)], ignore_index=True)
                if negfeatures
                else train
            )
            model = fit(augmented, columns, rounds)
            pred = model.predict(val[columns].astype(float), num_threads=4)
            metrics = evaluate_frame(val, pred)
            for m in metrics:
                m.update(setup=name, fold=fold)
            allmetrics.extend(metrics)
            for rid, p in zip(val["row_id"], pred, strict=True):
                oof.append(
                    dict(row_id=int(rid), setup=name, prediction=float(p), fold=fold)
                )
            foldstats[-1][name + "_generated_negatives"] = len(negrows)
            print(
                f"Fold {fold+1}/{folds} {name}: groups={len(metrics)} negatives={len(negrows)}",
                flush=True,
            )
    summaries = []
    for name, columns, mode in [("random", [], "none")] + setups:
        ms = [m for m in allmetrics if m["setup"] == name]
        summary = dict(
            setup=name,
            features=len(columns),
            groups=len(ms),
            ndcg_at_6=float(np.mean([m["ndcg_at_6"] for m in ms])),
            pairwise_accuracy=float(np.mean([m["pairwise_accuracy"] for m in ms])),
        )
        for family_name, prefix in [("manual", "manual"), ("crum_order", "crum")]:
            selected = [m for m in ms if ("|" + prefix) in m["group"]]
            summary[family_name + "_groups"] = len(selected)
            summary[family_name + "_ndcg_at_6"] = (
                float(np.mean([m["ndcg_at_6"] for m in selected])) if selected else None
            )
        for subset in ["manual", "manual_acceptance"]:
            selected = [m for m in ms if m["group"].endswith("|" + subset)]
            key = "manual_order" if subset == "manual" else subset
            summary[key + "_groups"] = len(selected)
            summary[key + "_ndcg_at_6"] = (
                float(np.mean([m["ndcg_at_6"] for m in selected])) if selected else None
            )
        summaries.append(summary)
    # Paired uncertainty for the augmentation question; cluster by spelling family.
    bysetup = {
        name: {m["group"]: m for m in allmetrics if m["setup"] == name}
        for name, _, _ in setups
    }
    family_by_group = dict(zip(frame["group"], frame["lemma"].map(family)))
    comparisons = []
    baseline = bysetup["gdex_ud_sm"]
    for name in ["naive_homograph_negatives", "conservative_barley_father_negatives"]:
        clusters = defaultdict(list)
        for group, b in baseline.items():
            clusters[family_by_group[group]].append(
                bysetup[name][group]["ndcg_at_6"] - b["ndcg_at_6"]
            )
        values = list(clusters.values())
        rng = np.random.default_rng(42)
        boot = []
        for _ in range(1000):
            sampled = [
                v for i in rng.integers(0, len(values), len(values)) for v in values[i]
            ]
            boot.append(float(np.mean(sampled)))
        comparisons.append(
            dict(
                setup=name,
                reference="gdex_ud_sm",
                delta_ndcg=float(np.mean([v for c in values for v in c])),
                cluster_bootstrap_95_ci=[
                    float(x) for x in np.quantile(boot, [0.025, 0.975])
                ],
                lemma_clusters=len(values),
            )
        )
    result = dict(
        protocol=dict(
            folds=folds,
            seed=42,
            rounds=rounds,
            split="lemma family, with cross-fold shared Coptic sentence purging",
            labels="manual order priority 1/2/3 -> 30/20/10; separate manual_acceptance query: selected ->10, explicit forbidden ->0; unknown excluded",
            gain="linear relevance 0..30; no clipping signed negatives into gold positives",
            crum="Crum examples are positive selections; order is learned within each TLA/sense group.",
            frequencies="fixed unlabeled full-corpus surface token frequencies, same for fit and inference",
            negatives="training folds only; one per positive at most, same target spelling family; all setups use identical validation candidates",
            warning="Scores evaluate preference among labeled candidates, not precision of unseen whole-corpus top-10",
        ),
        summaries=summaries,
        augmentation_comparisons=comparisons,
        folds=foldstats,
        groups=allmetrics,
    )
    write_json(config.reports / "ranking_experiment.json", result)
    write_json(config.reports / "out_of_fold_predictions.json", oof)
    # Final model uses all clean labels. Keep conservative pair augmentation optional rather than selecting on anecdotal anchors.
    final = fit(frame, GDEX + UD + SM, rounds)
    models = config.work / "models"
    models.mkdir(exist_ok=True)
    final.save_model(str(models / "ranker.txt"))
    with Corpus(config).connect() as db:
        stamp = db.execute(
            "SELECT value FROM metadata WHERE key='fingerprint'"
        ).fetchone()[0]
    write_json(
        models / "metadata.json",
        dict(
            features=GDEX + UD + SM,
            model=MODEL,
            training_rows=len(frame),
            rounds=rounds,
            augmentation="none; experimental modes remain ablations",
            corpus_fingerprint=stamp,
            feature_sha256=hashlib.sha256(
                (config.derived / "features.tsv").read_bytes()
            ).hexdigest(),
        ),
    )
    md = [
        "# 重建后的排序与负例实验",
        "",
        json.dumps(result["protocol"], ensure_ascii=False, indent=2),
        "",
        "| 设置 | 特征数 | 可评估 query | NDCG@6 | 成对顺序准确率 | 人工 NDCG | Crum 弱标签 NDCG |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        md.append(
            f"| {s['setup']} | {s['features']} | {s['groups']} | {s['ndcg_at_6']:.4f} | {s['pairwise_accuracy']:.4f} | {s['manual_ndcg_at_6']:.4f} | {s['crum_order_ndcg_at_6']:.4f} |"
        )
    md += ["", "## 负例增益的不确定性", ""]
    for c in comparisons:
        lo, hi = c["cluster_bootstrap_95_ci"]
        md.append(
            f"- {c['setup']} 相对同特征无负例：NDCG 差 {c['delta_ndcg']:+.5f}；按 lemma 聚类 bootstrap 95% 区间 [{lo:+.5f}, {hi:+.5f}]。"
        )
    md += [
        "",
        "Crum 相邻引用的 relevance 差很小，且许多 query 只有少量例句，随机排序也可能有很高 NDCG。应同时看 random、成对顺序准确率和人工子集；不能把 0.96 解读成“96% 的全语料例句都正确”。人工可评估 query 很少，子集结果不稳定。",
    ]
    md += [
        "",
        "人工选中但无 priority 的记录参与单独的“收录/禁用”二元任务，不被捏造成 priority=1。人工严格顺序与收录判断的分项结果保存在 JSON 中。",
    ]
    md += [
        "",
        "本表为本轮真实运行结果。不同模型使用相同验证候选；负例只在训练折中产生。C992/C998 同一词形在同一验证折，不能通过把 father 放进训练、barley 放进验证来伪造泛化。",
        "人工与 Crum 的排名 query 分开；没有把不同来源的任意数值当作同一条严格偏好序列。Crum 引用顺序受词义、词形、方言和文献组织影响，尚不等于例句教学质量真值。",
        "导出的部署模型在全部清洁训练数据上重新训练；不是复用最后一折。默认不加入推导负例，负例实验与基本可用模型分开保留。",
        "这些 NDCG 与旧项目的 0.8128 使用不同清洁数据、分组、验证及增益协议，不能直接比较大小。全语料 top-10 仍需独立人工标注。",
    ]
    (config.reports / "RANKING_EXPERIMENT.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    return summaries
