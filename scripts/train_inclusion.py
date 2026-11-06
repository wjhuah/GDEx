#!/usr/bin/env python3
"""TAB inclusion with Crum positives and same-form hard negatives; matched tests."""

import sys, json, hashlib
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config, write_json
from coptic_examples.lexicon import load
from coptic_examples.inclusion import EntryFeatures, COLUMNS_INCLUSION
from coptic_examples.ablation import evaluate, summary, difference
from coptic_examples.sense import targets, align_candidate

cfg = Config()
lex = load(cfg.lexicon)
raw = [json.loads(l) for l in (cfg.derived / "ablation_rows.jsonl").open()]
reports = []


def fit(items, columns):
    y = np.array([r["y"] for r in items])
    w = np.array([r["weight"] for r in items])
    s = [w[y == i].sum() for i in [0, 1]]
    if not all(s):
        return None
    w *= np.array([sum(s) / (2 * s[i]) for i in y])
    ds = lgb.Dataset(
        np.array([r["features"] for r in items])[:, columns],
        label=y,
        weight=w,
        feature_name=[COLUMNS_INCLUSION[i] for i in columns],
    )
    return lgb.train(
        dict(
            objective="binary",
            num_leaves=15,
            min_data_in_leaf=20,
            learning_rate=0.05,
            lambda_l2=2.0,
            verbosity=-1,
            num_threads=4,
            seed=42,
            deterministic=True,
            force_col_wise=True,
        ),
        ds,
        num_boost_round=120,
    )


def fit_order(items):
    groups = defaultdict(list)
    for r in items:
        groups[r["group"]].append(r)
    ordered = []
    gains = []
    weights = []
    sizes = []
    for group, values in sorted(groups.items()):
        if len({r["gain"] for r in values}) < 2:
            continue
        ordered.extend(values)
        gains.extend(round(r["gain"], 8) for r in values)
        weights.extend([1 / len(values)] * len(values))
        sizes.append(len(values))
    if not ordered:
        return None
    levels = sorted(set(gains))
    labels = [levels.index(v) for v in gains]
    ds = lgb.Dataset(
        np.asarray([r["features"] for r in ordered]),
        label=labels,
        weight=weights,
        group=sizes,
        feature_name=COLUMNS_INCLUSION,
    )
    return lgb.train(
        dict(
            objective="lambdarank",
            label_gain=levels,
            num_leaves=15,
            min_data_in_leaf=15,
            learning_rate=0.05,
            lambda_l2=2.0,
            verbosity=-1,
            num_threads=4,
            seed=42,
            deterministic=True,
            force_col_wise=True,
        ),
        ds,
        num_boost_round=120,
    )


from coptic_examples.dialects import Dialects

for dialect in Dialects().data:
    data = [r for r in raw if r["dialect"] == dialect]
    if not data:
        continue
    ef = EntryFeatures(cfg, lex, dialect)
    manual = []
    crum = []
    known = set()
    positive_by_text = defaultdict(set)
    for r in data:
        if r["task"] == "acceptance" and r["gain"] > 0 and r["TLA"]:
            positive_by_text[r["text_key"]].add(r["TLA"])
    for r in data:
        if r["task"] == "crum_style" and r["TLA"]:
            positive_by_text[r["text_key"]].add(r["TLA"])
    for r in data:
        if r["task"] not in {"acceptance", "crum_style"} or not r["TLA"]:
            continue
        e = lex[r["TLA"]]
        key = (e.tla, r["text_key"])
        if key in known:
            continue
        if not align_candidate(r["candidate"], e, allow_pos_mismatch=True):
            continue
        known.add(key)
        item = dict(
            r,
            features=ef.vector(r["candidate"], e, base=r["x"][:44]),
            y=int(r["gain"] > 0) if r["task"] == "acceptance" else 1,
            weight=1.0,
        )
        (manual if r["task"] == "acceptance" else crum).append(item)
    preference_rows = [
        dict(r, features=ef.vector(r["candidate"], lex[r["TLA"]], base=r["x"][:44]))
        for r in data
        if r["TLA"]
        and align_candidate(r["candidate"], lex[r["TLA"]], allow_pos_mismatch=True)
    ]
    families = sorted({r["family"] for r in data})
    np.random.default_rng(42).shuffle(families)
    fm = {v: i % 5 for i, v in enumerate(families)}

    def negatives(train):
        existing = {(r["TLA"], r["text_key"]) for r in train}
        out = []
        counts = Counter()
        for r in sorted(train, key=lambda r: (r["TLA"], r["text_key"])):
            if not r["y"]:
                continue
            own = lex[r["TLA"]]
            for e in ef.rivals(own):
                if (
                    e.lemma != own.lemma
                    or e.pos != own.pos
                    or (e.tla, r["text_key"]) in existing
                    or e.tla in positive_by_text[r["text_key"]]
                    or counts[e.tla] >= 20
                ):
                    continue
                x = ef.vector(r["candidate"], e)
                # Source and target must differ in explicit lexical support; ambiguous spans stay unlabelled.
                if not targets(r["candidate"], e):
                    continue
                own_cos = r["features"][45]
                other_cos = x[45]
                semantic_support = (
                    np.isfinite(own_cos)
                    and np.isfinite(other_cos)
                    and own_cos > other_cos
                )
                if r["features"][44] <= x[44] and not semantic_support:
                    continue
                out.append(
                    dict(
                        r,
                        TLA=e.tla,
                        gloss=e.gloss,
                        features=x,
                        y=0,
                        weight=0.2,
                        source_tla=r["TLA"],
                        synthetic=True,
                    )
                )
                existing.add((e.tla, r["text_key"]))
                counts[e.tla] += 1
        return out

    setups = {
        "tab_sentence_only": (list(range(44)), False, False),
        "tab_conditioned": (list(range(len(COLUMNS_INCLUSION))), False, False),
        "tab_crum_conditioned": (list(range(len(COLUMNS_INCLUSION))), True, False),
        "tab_crum_hard_negatives": (list(range(len(COLUMNS_INCLUSION))), True, True),
        "tab_crum_without_rival_count": (
            [i for i, c in enumerate(COLUMNS_INCLUSION) if c != "rival_count"],
            True,
            False,
        ),
        "tab_crum_without_typicality": (
            [
                i
                for i, c in enumerate(COLUMNS_INCLUSION)
                if c
                not in {
                    "frame_log_frequency",
                    "frame_probability",
                    "collocation_log_mean",
                    "collocation_log_max",
                }
            ],
            True,
            False,
        ),
    }
    metrics = {k: [] for k in setups}
    fold_records = []
    cv = []
    order_validation = []
    fusion = defaultdict(list)
    for fold in range(5):
        val = [r for r in manual if fm[r["family"]] == fold]
        held = {r["text_key"] for r in data if fm[r["family"]] == fold}
        mt = [
            r for r in manual if fm[r["family"]] != fold and r["text_key"] not in held
        ]
        wt = [r for r in crum if fm[r["family"]] != fold and r["text_key"] not in held]
        if len({r["y"] for r in val}) < 2 or len({r["y"] for r in mt}) < 2:
            continue
        fold_models = {}
        for name, (cols, usecrum, useneg) in setups.items():
            tr = mt + (wt if usecrum else [])
            neg = negatives(tr) if useneg else []
            assert not {r["text_key"] for r in tr + neg} & held
            model = fit(tr + neg, cols)
            p = model.predict(
                np.array([r["features"] for r in val])[:, cols], num_threads=4
            )
            ms = evaluate(val, p)
            metrics[name].extend(ms)
            if name in {"tab_crum_conditioned", "tab_crum_hard_negatives"}:
                fold_models[name] = model
            cv.append(
                dict(
                    fold=fold,
                    setup=name,
                    auc=float(roc_auc_score([r["y"] for r in val], p)),
                    average_precision=float(
                        average_precision_score([r["y"] for r in val], p)
                    ),
                    manual_train=len(mt),
                    crum_train=len(wt) if usecrum else 0,
                    synthetic_train=len(neg),
                    validation=len(val),
                )
            )
        ot = [
            r
            for r in preference_rows
            if fm[r["family"]] != fold and r["text_key"] not in held
        ]
        ov = [r for r in preference_rows if fm[r["family"]] == fold]
        om = fit_order(ot)
        if om and ov:
            matrix = np.asarray([r["features"] for r in ov])
            raw_order = om.predict(matrix, num_threads=4)
            order_validation.extend(evaluate(ov, raw_order))
            order_score = 1 / (1 + np.exp(-np.clip(raw_order, -30, 30)))
            for setup, im in fold_models.items():
                inc = im.predict(matrix, num_threads=4)
                for alpha in [0.0, 0.2, 0.5, 1.0]:
                    fusion[(setup, alpha)].extend(
                        evaluate(ov, (1 - alpha) * inc + alpha * order_score)
                    )
        print(dialect, "fold", fold, "complete", flush=True)
    neg = negatives(manual + crum)
    # Primary product metric is within-entry ranking of known TAB candidates. Preserve augmentation as a measured alternative.
    selected = (
        max(
            ["tab_crum_conditioned", "tab_crum_hard_negatives"],
            key=lambda name: (
                summary(metrics[name])["acceptance"]["ndcg_at_6"]
                if summary(metrics[name])["acceptance"]["ndcg_at_6"] is not None
                else -1
            ),
        )
        if cv
        else "tab_crum_conditioned"
    )
    ranking_mix = 0.2
    if fusion:
        choices = {key: summary(value) for key, value in fusion.items()}
        eligible = {
            key: value
            for key, value in choices.items()
            if value["acceptance"]["ndcg_at_6"] is not None
            and value["crum_style"]["ndcg_at_6"] is not None
        }
        if eligible:
            selected, ranking_mix = max(
                eligible,
                key=lambda key: (
                    eligible[key]["acceptance"]["ndcg_at_6"]
                    + eligible[key]["crum_style"]["ndcg_at_6"]
                )
                / 2,
            )
    cols, usecrum, useneg = setups[selected]
    final = manual + (crum if usecrum else [])
    finalneg = negatives(final) if useneg else []
    model = fit(final + finalneg, cols)
    contrast = []
    if dialect == "sahidic":
        anchors = [r for r in manual if r["TLA"] in {"C992", "C998"} and r["y"]]
        for anchor in anchors:
            held_key = anchor["text_key"]
            held_translation = anchor["candidate"]["translation"]
            tr = [
                r
                for r in manual + crum
                if r["text_key"] != held_key
                and r["candidate"]["translation"] != held_translation
            ]
            other = lex[{"C992": "C998", "C998": "C992"}[anchor["TLA"]]]
            cross = ef.vector(anchor["candidate"], other)
            for mode in ["without_cross_negatives", "with_cross_negatives"]:
                nn = negatives(tr) if mode == "with_cross_negatives" else []
                m = fit(tr + nn, list(range(len(COLUMNS_INCLUSION))))
                pp = m.predict(np.array([anchor["features"], cross]), num_threads=4)
                contrast.append(
                    dict(
                        mode=mode,
                        own=anchor["TLA"],
                        cross=other.tla,
                        id=anchor["candidate"]["id"],
                        margin=float(pp[0] - pp[1]),
                        correct=bool(pp[0] > pp[1]),
                        synthetic_train=len(nn),
                    )
                )
    order_model = fit_order(preference_rows)
    if order_model:
        order_model.save_model(
            str(cfg.work / "models" / dialect / "preference_learned.txt")
        )
    if model:
        folder = cfg.work / "models" / dialect
        model.save_model(str(folder / "inclusion_learned.txt"))
        write_json(
            folder / "inclusion_learned.json",
            dict(
                file="inclusion_learned.txt",
                order_file="preference_learned.txt" if order_model else None,
                ranking_mix=ranking_mix,
                columns=COLUMNS_INCLUSION,
                objective="Dictionary-conditioned inclusion from manual labels; Crum positives and inferred cross negatives according to selected_variant",
                manual_rows=len(manual),
                crum_positive_rows=len(crum),
                synthetic_rows=len(finalneg),
                selected_variant=selected,
                selection_criterion="Mean of TAB inclusion and Crum ordering NDCG across five folds; exploratory same-fold selection",
                weights=dict(manual=1.0, crum=1.0, synthetic=0.2),
                provisional=len(manual) < 100,
                manual_sha256=hashlib.sha256(cfg.manual.read_bytes()).hexdigest(),
                evaluation="Exploratory matched family folds; no independent token-sense gold; raw score is uncalibrated",
            ),
        )
        gain = model.feature_importance("gain")
        split = model.feature_importance("split")
        importance = sorted(
            [
                dict(
                    feature=k,
                    gain_share=float(g / sum(gain)) if sum(gain) else 0.0,
                    split=int(n),
                )
                for k, g, n in zip(COLUMNS_INCLUSION, gain, split)
            ],
            key=lambda r: -r["gain_share"],
        )
    else:
        importance = []
    reports.append(
        dict(
            dialect=dialect,
            manual_rows=len(manual),
            crum_rows=len(crum),
            synthetic_rows=len(neg),
            selected_variant=selected,
            ranking_mix=ranking_mix,
            fusion_validation=[
                dict(setup=key[0], alpha=key[1], summary=summary(value))
                for key, value in fusion.items()
            ],
            order_validation=summary(order_validation),
            anchor_contrast=contrast,
            folds=cv,
            summaries={k: summary(v) for k, v in metrics.items()},
            comparisons={
                k: difference(v, metrics["tab_conditioned"], "acceptance")
                for k, v in metrics.items()
                if k != "tab_conditioned"
            },
            hard_negative_effect=difference(
                metrics["tab_crum_hard_negatives"],
                metrics["tab_crum_conditioned"],
                "acceptance",
            ),
            typicality_effect=difference(
                metrics["tab_crum_without_typicality"],
                metrics["tab_crum_conditioned"],
                "acceptance",
            ),
            rival_count_effect=difference(
                metrics["tab_crum_without_rival_count"],
                metrics["tab_crum_conditioned"],
                "acceptance",
            ),
            importance=importance,
            synthetic_examples=[
                dict(
                    target=r["TLA"],
                    source=r["source_tla"],
                    id=r["candidate"]["id"],
                    translation=r["candidate"]["translation"],
                )
                for r in neg[:20]
            ],
        )
    )
    ef.close()
write_json(
    cfg.reports / "importance_audit/inclusion_experiment.json",
    dict(
        protocol="New inclusion objective with training-only augmentation; manual validation candidates fixed across setups; family-grouped folds and text purging. Crum citations are positive examples; Crum editorial ranks are retained in the separate existing preference model. Cross labels are inferred and need token expert review.",
        results=reports,
    ),
)
