"""Learned sense comparisons at a single TT target token."""

import copy
import json
import numpy as np
import lightgbm as lgb
from .inclusion import COLUMNS_INCLUSION
from .sense import targets, align_candidate

ROLES = ["vocative", "nsubj", "obj", "nmod", "root", "appos"]
COLUMNS = COLUMNS_INCLUSION + ["entry_title"] + ["target_role_" + r for r in ROLES]


def vector(features, candidate, entry):
    ii = targets(candidate, entry)
    return (
        features.vector(candidate, entry)
        + [float("Titel" in entry.subcategories)]
        + [
            float(any(candidate["tokens"][i]["dep"].split(":")[0] == role for i in ii))
            for role in ROLES
        ]
    )


def dataset(rows, features, lex):
    known = {(r["tla"], r["text_key"]) for r in rows if r["y"] > 0}
    out = {}
    allowed = {r["tla"] for r in rows}
    for r in rows:
        if r["y"] < 0:
            continue
        source = copy.deepcopy(r["candidate"])
        own = align_candidate(source, lex[r["tla"]], True)
        if len(own) != 1:
            continue
        rivals = []
        for e in features.rivals(lex[r["tla"]]):
            if e.tla not in allowed:
                continue
            if (e.tla, r["text_key"]) in known:
                continue
            c = copy.deepcopy(source)
            ii = align_candidate(c, e, True)
            if ii != own:
                continue
            rivals.append((e, c))
        if not rivals:
            continue
        out[(r["tla"], r["text_key"])] = dict(
            r, y=1, x=vector(features, source, lex[r["tla"]])
        )
        for e, c in rivals:
            key = (e.tla, r["text_key"])
            if key not in known:
                out[key] = dict(r, tla=e.tla, y=0, x=vector(features, c, e))
    return list(out.values())


def fit(rows):
    y = np.asarray([r["y"] for r in rows])
    counts = np.bincount(y, minlength=2)
    if not all(counts):
        return None
    weights = np.asarray([len(rows) / (2 * counts[v]) for v in y])
    data = lgb.Dataset(
        np.asarray([r["x"] for r in rows]),
        label=y,
        weight=weights,
        feature_name=COLUMNS,
    )
    return lgb.train(
        dict(
            objective="binary",
            num_leaves=31,
            min_data_in_leaf=1,
            learning_rate=0.05,
            lambda_l2=0.1,
            verbosity=-1,
            num_threads=4,
            seed=42,
            deterministic=True,
            force_col_wise=True,
        ),
        data,
        num_boost_round=400,
    )


def resolve(model, rows, entry, lex, architecture):
    path = model.folder / "token_sense.txt"
    if not path.exists():
        return None
    metadata_path = model.folder / "token_sense.json"
    if not metadata_path.exists():
        raise RuntimeError("Rebuild token-sense model metadata")
    metadata = json.loads(metadata_path.read_text())
    for key in ["corpus_fingerprint", "lexicon_sha256", "manual_sha256"]:
        if metadata.get(key) != model.metadata.get(key):
            raise RuntimeError("Retrain token-sense model after source changes")
    if not metadata.get("available"):
        return None
    scorer = lgb.Booster(model_file=str(path))
    out = []
    for row in rows:
        matching = targets(row, entry)
        winners = []
        scores = []
        for i in matching:
            tid = row["tokens"][i]["id"]
            entries = [entry]
            probes = []
            for rival in model.features.rivals(entry):
                c = copy.deepcopy(row)
                if i in align_candidate(c, rival, True):
                    entries.append(rival)
            if len(entries) == 1:
                winners.append(tid)
                continue
            for e in entries:
                c = copy.deepcopy(row)
                c.setdefault("target_alignment", {})[e.tla] = [tid]
                probes.append(vector(model.features, c, e))
            p = scorer.predict(np.asarray(probes), num_threads=4)
            scores.append(
                dict(
                    token_id=tid,
                    predictions={e.tla: float(v) for e, v in zip(entries, p)},
                )
            )
            if p[0] > 0.5 and (len(p) == 1 or p[0] > max(p[1:]) + 1e-10):
                winners.append(tid)
        if not winners:
            continue
        row.setdefault("target_alignment", {})[entry.tla] = winners
        row["target_token_evidence"] = [t for t in row["tokens"] if t["id"] in winners]
        row["token_sense_predictions"] = scores
        out.append(row)
    if out:
        inclusion, rank = model.score(out, entry, architecture)
        retained = []
        for row, p, s in zip(out, inclusion, rank):
            if p <= model.metadata.get("thresholds", {}).get(architecture, 0.5):
                continue
            retained.append(row)
            row.update(
                learned_inclusion_score=float(p),
                shortlist_score=float(s),
                score=float(s),
                acceptance_score=float(p),
                quality_score=float(p),
            )
    return retained if out else []
