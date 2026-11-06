"""Matched-candidate ablations; Crum style and expert acceptance stay separate.

Only cached, real BGE vectors are read. No downloads, embeddings or LLM calls.
All generated negatives are training-only, after spelling-family/text purging.
"""

import hashlib
import importlib.util
import inspect
import json
import math
import re
import sys
from collections import Counter, defaultdict
import numpy as np
import lightgbm as lgb
from .config import database, write_json
from .corpus import Corpus
from .dataset import read_rows, text_key
from .dialects import Dialects
from .features import Features, GDEX, UD
from .lexicon import load
from .shortlist import INFO, COLUMNS, numbers
from .selection import profile
from .text import norm, lemma_key, real_translation, editorial_heading, sm_freq

CHEAP = COLUMNS + ["sense_overlap"]
SEMANTIC = CHEAP + ["sense_cosine"]
EXTRA_POS = (
    "VERB AUX NOUN PRON ADJ ADV ADP SCONJ CCONJ DET NUM PART PROPN INTJ X".split()
)
EXTRA_DEP = "nsubj nsubj:pass obj iobj obl ccomp xcomp advcl acl acl:relcl csubj cc conj aux mark det case nmod".split()
EXTRA = (
    [f"ud_pos_{v}" for v in EXTRA_POS]
    + [f'ud_dep_{v.replace(":","_")}' for v in EXTRA_DEP]
    + ["ud_tree_width", "ud_branching_nodes", "ud_max_dep_distance"]
)
ALL_COLUMNS = SEMANTIC + EXTRA


def feature_fingerprint(config):
    from .sense import targets, NATIVE, align_candidate

    paths = [
        config.manual,
        config.lexicon,
        config.derived / "crum_bible_occurrences.jsonl",
        config.vectors,
        config.source / "crum3/deploy/crum4/retrieval.py",
        config.source / "crum3/deploy/crum4/cache/corpus.sqlite3",
    ]
    paths += [config.work / "models" / d / "frequencies.json" for d in Dialects().data]
    with Corpus(config).connect() as db:
        corpus = db.execute(
            "SELECT value FROM metadata WHERE key='fingerprint'"
        ).fetchone()[0]
    code = "\n".join(
        inspect.getsource(v)
        for v in [
            build_rows,
            extended_ud,
            numbers,
            Features,
            profile,
            CachedVectors,
            targets,
        ]
    ) + json.dumps(NATIVE, default=lambda v: sorted(v), sort_keys=True)
    return dict(
        corpus=corpus,
        features=hashlib.sha256(code.encode()).hexdigest(),
        inputs={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
            if p.exists()
        },
    )


def extended_ud(candidate):
    ts = candidate["tokens"]
    pos = Counter(t["pos"] for t in ts)
    dep = Counter(t["dep"] for t in ts)
    byid = {t["id"]: t for t in ts}
    order = {t["id"]: i for i, t in enumerate(ts)}
    children = Counter(t["head"] for t in ts if t["head"] in byid)
    levels = Counter()
    for t in ts:
        node = t
        seen = set()
        while node["head"] in byid and node["id"] not in seen:
            seen.add(node["id"])
            node = byid[node["head"]]
        levels[len(seen)] += 1
    return (
        [pos[v] for v in EXTRA_POS]
        + [dep[v] for v in EXTRA_DEP]
        + [
            max(levels.values(), default=0),
            sum(v > 1 for v in children.values()),
            max(
                (
                    abs(order[t["id"]] - order[t["head"]])
                    for t in ts
                    if t["head"] in order
                ),
                default=0,
            ),
        ]
    )


def family(lemma):
    return re.sub(r"([ⲁⲉⲏⲓⲟⲱⲩ])\1", r"\1", lemma_key(lemma))


def crum4_module(config):
    name = "audited_legacy_crum4"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, config.source / "crum3/deploy/crum4/retrieval.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class CachedVectors:
    """Never substitute a zero/lexical vector when a cached BGE vector is absent."""

    def __init__(self, config):
        self.config = config
        config.vectors.parent.mkdir(parents=True, exist_ok=True)
        self.connection = database(config.vectors)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS vector(model TEXT,role TEXT,key TEXT,payload TEXT,PRIMARY KEY(model,role,key))"
        )
        self.connection.commit()
        self.memo = {}
        self.extractors = {}

    def get(self, text, role):
        key = (role, norm(text))
        if key not in self.memo:
            digest = hashlib.sha256(key[1].encode()).hexdigest()
            row = self.connection.execute(
                "SELECT payload FROM vector WHERE model=? AND role=? AND key=?",
                ("BAAI/bge-large-en-v1.5", role, digest),
            ).fetchone()
            self.memo[key] = (
                np.asarray(json.loads(row[0]), dtype=np.float32) if row else None
            )
        return self.memo[key]

    def cosine(self, gloss, translation):
        left, right = self.get(gloss, "definition"), self.get(translation, "sentence")
        return float(left @ right) if left is not None and right is not None else None

    def target_features(self, candidate, entry, dialect):
        if dialect not in self.extractors:
            freq = json.loads(
                (self.config.work / "models" / dialect / "frequencies.json").read_text()
            )
            self.extractors[dialect] = Features(freq)
        x, _, _ = numbers(candidate, entry, self.extractors[dialect])
        return x + [sm_freq(entry.gloss, candidate["translation"])]


def build_rows(config):
    corpus = Corpus(config)
    lex = load(config.lexicon)
    registry = Dialects()
    vectors = CachedVectors(config)
    from .text import rouge
    from .features import STOP

    legacy_index = config.source / "crum3/deploy/crum4/cache/corpus.sqlite3"
    oldfreq = {}
    if legacy_index.exists():
        with database(legacy_index) as db:
            oldfreq = dict(
                db.execute("SELECT lemma_norm, frequency FROM token_frequency")
            )
    extractors = {
        d: Features(
            json.loads((config.work / "models" / d / "frequencies.json").read_text())
        )
        for d in registry.data
    }
    result = []
    memo = {}
    seen = set()

    def attributes(c, e, dialect):
        from .sense import align_candidate

        align_candidate(c, e, allow_pos_mismatch=True)
        key = (c["id"], e.tla, dialect)
        if key not in memo:
            x, syn, heuristic = numbers(c, e, extractors[dialect])
            words = [
                t["lemma"] for t in c["tokens"] if t["lemma"] and t["lemma"] not in STOP
            ]
            memo[key] = dict(
                x=x + [sm_freq(e.gloss, c["translation"])],
                extra=extended_ud(c),
                cosine=vectors.cosine(e.gloss, c["translation"]),
                heuristic=heuristic,
                rouge=rouge(e.gloss, c["translation"]),
                old_frequency=sum(oldfreq.get(w, 0) for w in words)
                / max(1, len(words)),
            )
        return memo[key]

    for r in read_rows(config):
        if (
            r["evidence"] not in {"manual", "manual_forbidden_rule"}
            or r.get("dialect_id") not in registry.data
            or r["TLA"] not in lex
        ):
            continue
        c = r["candidate"]
        e = lex[r["TLA"]]
        d = r["dialect_id"]
        if not real_translation(c["translation"]) or editorial_heading(
            c["translation"]
        ):
            continue
        base = dict(
            TLA=e.tla,
            lemma=e.lemma,
            family=family(e.lemma),
            dialect=d,
            text_key=text_key(c),
            candidate=c,
            gloss=e.gloss,
            **attributes(c, e, d),
        )
        key = (e.tla, d, base["text_key"])
        if key in seen:
            continue
        seen.add(key)
        result.append(
            dict(
                base,
                task="acceptance",
                group=e.tla + "|acceptance|" + d,
                gain=float(r["evidence"] == "manual"),
                priority=r.get("priority"),
                rank=None,
            )
        )
        if r["evidence"] == "manual" and r.get("priority") is not None:
            result.append(
                dict(
                    base,
                    task="manual_order",
                    group=e.tla + "|manual_order|" + d,
                    gain=(4 - r["priority"]) / 3,
                    priority=r["priority"],
                    rank=r["priority"],
                )
            )
    with (
        database(config.work / "cache/crum_bible.sqlite3") as audit,
        corpus.connect() as db,
    ):
        seen = set()
        for d in registry.data:
            query = "SELECT o.payload,l.sentence_id FROM occurrence o JOIN link l ON o.id=l.occurrence_id WHERE l.dialect=? AND l.status='verified_target_form' ORDER BY o.file,o.article,o.editorial_order,l.sentence_id"
            for payload, sid in audit.execute(query, (d,)):
                r = json.loads(payload)
                c = json.loads(
                    db.execute(
                        "SELECT payload FROM sentence WHERE id=?", (sid,)
                    ).fetchone()[0]
                )
                if not real_translation(c["translation"]) or editorial_heading(
                    c["translation"]
                ):
                    continue
                possible = []
                for tla in r["tla_candidates"]:
                    e = lex[tla]
                    codes = {
                        code
                        for values in (e.form_dialects or {}).values()
                        for code in values
                    }
                    if codes and d not in registry.from_codes(codes):
                        continue
                    if profile(c, e.lemma, e.forms)["target_indices"]:
                        possible.append(tla)
                if (
                    not possible
                    or len({(lex[t].lemma, lex[t].pos, lex[t].gloss) for t in possible})
                    != 1
                ):
                    continue
                # Semantic comparisons require the same gloss, not just lemma/POS.
                e = lex[possible[0]]
                key = (d, r["article_id"], text_key(c))
                if key in seen:
                    continue
                seen.add(key)
                result.append(
                    dict(
                        TLA=e.tla if len(possible) == 1 else None,
                        tla_candidates=possible,
                        lemma=e.lemma,
                        family=family(e.lemma),
                        dialect=d,
                        text_key=text_key(c),
                        candidate=c,
                        gloss=e.gloss,
                        task="crum_style",
                        group=r["article_id"] + "|" + d,
                        rank=r["editorial_order"],
                        priority=None,
                        occurrence_id=r["id"],
                        **attributes(c, e, d),
                    )
                )
    groups = defaultdict(list)
    for r in result:
        if r["task"] == "crum_style":
            groups[r["group"]].append(r)
    for group, items in groups.items():
        levels = sorted({r["rank"] for r in items})
        for r in items:
            r["gain"] = 1 - 0.8 * levels.index(r["rank"]) / max(1, len(levels) - 1)
    vectors.connection.close()
    path = config.derived / "ablation_rows.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(result):
            r["row_id"] = i
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_json(
        config.derived / "ablation_fingerprint.json", feature_fingerprint(config)
    )
    write_json(
        config.reports / "ablation_inventory.json",
        dict(
            rows=len(result),
            by_dialect_task=dict(
                Counter(r["dialect"] + "/" + r["task"] for r in result)
            ),
            cached_BGE_rows=sum(r["cosine"] is not None for r in result),
            note="Stricter than style-only training: ambiguous TLA candidates must also share the identical English gloss for semantic comparisons.",
        ),
    )
    print(
        "Ablation features built:",
        len(result),
        "cached semantic rows:",
        sum(r["cosine"] is not None for r in result),
        flush=True,
    )
    return result


def train_gain(row, scheme):
    if scheme == "continuous":
        return row["gain"]
    if row["task"] == "crum_style":
        return max(1.0, 3.0 - 0.1 * (row["rank"] - 1)) / 3
    return row["gain"]  # TAB keeps 3/2/1; unranked examples stay a binary task.


def fit_ranker(rows, columns, scheme="continuous", rounds=100):
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    ordered = []
    sizes = []
    weights = []
    values = []
    for group, items in sorted(groups.items()):
        if len({train_gain(r, scheme) for r in items}) < 2:
            continue
        ordered.extend(items)
        sizes.append(len(items))
        weights.extend(
            [(0.35 if r["task"] == "crum_style" else 1.0) / len(items) for r in items]
        )
        values.extend(round(train_gain(r, scheme), 8) for r in items)
    if not ordered:
        return None
    gains = sorted(set(values))
    indices = [ALL_COLUMNS.index(c) for c in columns]
    matrix = np.asarray(
        [
            r["x"] + [r["cosine"] if r["cosine"] is not None else 0.0] + r["extra"]
            for r in ordered
        ]
    )[:, indices]
    # The cosine column is never selected for a row with a missing vector.
    if "sense_cosine" in columns:
        assert all(r["cosine"] is not None for r in ordered)
    labels = [gains.index(v) for v in values]
    data = lgb.Dataset(
        matrix, label=labels, weight=weights, group=sizes, feature_name=columns
    )
    model = lgb.train(
        dict(
            objective="lambdarank",
            label_gain=gains,
            learning_rate=0.05,
            num_leaves=15,
            min_data_in_leaf=15,
            lambda_l2=1.0,
            verbosity=-1,
            num_threads=4,
            seed=42,
            deterministic=True,
            force_col_wise=True,
            lambdarank_truncation_level=10,
        ),
        data,
        num_boost_round=rounds,
    )
    return model


def predict(model, rows, columns):
    indices = [ALL_COLUMNS.index(c) for c in columns]
    return (
        model.predict(
            np.asarray(
                [
                    r["x"]
                    + [r["cosine"] if r["cosine"] is not None else 0.0]
                    + r["extra"]
                    for r in rows
                ]
            )[:, indices],
            num_threads=4,
        )
        if model
        else np.zeros(len(rows))
    )


def quality_first(train, val, joint_scores, two_head=False):
    from .shortlist import tree_acceptance, decision, pair_data, fit_linear

    accepted = [
        dict(r, x=r["x"][: len(COLUMNS)], y=int(r["gain"] > 0))
        for r in train
        if r["task"] == "acceptance"
    ]
    if len({r["y"] for r in accepted}) < 2:
        acceptance = np.full(len(val), 0.5)
    else:
        model = tree_acceptance(accepted)
        acceptance = 1 / (
            1
            + np.exp(
                -np.clip(
                    decision(model, [r["x"][: len(COLUMNS)] for r in val]), -30, 30
                )
            )
        )
    info = np.asarray([r["x"][COLUMNS.index("information_score")] for r in val])
    if not two_head:
        return (
            0.60 * acceptance
            + 0.25 / (1 + np.exp(-np.clip(joint_scores, -30, 30)))
            + 0.15 * info
        )
    style = [
        dict(r, x=r["x"][: len(COLUMNS)]) for r in train if r["task"] == "crum_style"
    ]
    x, y = pair_data(style)
    raw = (
        decision(fit_linear(x, y, False), [r["x"][: len(COLUMNS)] for r in val])
        if len(y)
        else np.zeros(len(val))
    )
    groups = defaultdict(list)
    for i, r in enumerate(val):
        groups[r["group"]].append(i)
    percentiles = np.full(len(val), 0.5)
    for indices in groups.values():
        values = raw[indices]
        # Ties use their average rank, not a hidden source-order preference.
        for i, value in zip(indices, values):
            percentiles[i] = (
                np.sum(values < value) + 0.5 * (np.sum(values == value) - 1)
            ) / max(1, len(values) - 1)
    return (
        0.60 * acceptance
        + 0.20 * percentiles
        + 0.15 * info
        + 0.05 * np.asarray([r["x"][-1] for r in val])
    )


def expected_top_metric(gains, scores, k, discounted=True):
    """Expected DCG/precision under uniformly random permutations within ties."""
    gains = np.asarray(gains, dtype=float)
    scores = np.asarray(scores)
    order = np.argsort(-scores, kind="stable")
    total = 0.0
    start = 0
    k = min(k, len(gains))
    while start < k:
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        mean = float(np.mean(gains[order[start:end]]))
        total += sum(
            mean / math.log2(i + 2) if discounted else mean / k
            for i in range(start, min(k, end))
        )
        start = end
    if not discounted:
        return total
    ideal = sum(
        v / math.log2(i + 2) for i, v in enumerate(sorted(gains, reverse=True)[:k])
    )
    return total / ideal if ideal else None


def evaluate(rows, scores):
    from .training import pair_accuracy

    groups = defaultdict(list)
    out = []
    for r, p in zip(rows, scores, strict=True):
        groups[r["group"]].append((r, float(p)))
    for group, items in sorted(groups.items()):
        gains = [r["gain"] for r, p in items]
        pred = [p for r, p in items]
        if len(set(gains)) < 2:
            continue
        positives = [r["gain"] > 0 for r, p in items]
        record = dict(
            group=group,
            family=items[0][0]["family"],
            task=items[0][0]["task"],
            n=len(items),
            ndcg_at_6=expected_top_metric(gains, pred, 6),
            pair_accuracy=pair_accuracy(gains, pred),
        )
        if record["task"] == "acceptance":
            record["known_precision_at_3"] = expected_top_metric(
                positives, pred, 3, False
            )
        out.append(record)
    return out


def generate_negatives(rows, mode, lex, vectors):
    if mode == "none":
        return []
    selected = [r for r in rows if r["task"] == "acceptance" and r["gain"] > 0]
    targets = {r["TLA"]: r for r in selected if r["TLA"]}
    existing = {(r["TLA"], r["text_key"]) for r in rows}
    generated = []
    for tla, target in sorted(targets.items()):
        count = 0
        for source in selected:
            if (
                source["family"] != target["family"]
                or source["TLA"] == tla
                or (tla, source["text_key"]) in existing
            ):
                continue
            if mode == "barley_father":
                if {tla, source["TLA"]} != {"C992", "C998"}:
                    continue
                words = set(
                    re.findall("[a-z]+", source["candidate"]["translation"].lower())
                )
                own = bool(
                    words
                    & (
                        {"barley"}
                        if source["TLA"] == "C998"
                        else {
                            "father",
                            "fathers",
                            "parent",
                            "parents",
                            "ancestor",
                            "ancestors",
                        }
                    )
                )
                other = bool(
                    words
                    & (
                        {"barley"}
                        if tla == "C998"
                        else {
                            "father",
                            "fathers",
                            "parent",
                            "parents",
                            "ancestor",
                            "ancestors",
                        }
                    )
                )
                if not own or other:
                    continue
            entry = lex[tla]
            x = vectors.target_features(source["candidate"], entry, target["dialect"])
            cosine = vectors.cosine(entry.gloss, source["candidate"]["translation"])
            generated.append(
                dict(
                    source,
                    TLA=tla,
                    gloss=entry.gloss,
                    group=target["group"],
                    gain=0.0,
                    x=x,
                    cosine=cosine,
                    synthetic=True,
                    source_tla=source["TLA"],
                    row_id=-1,
                )
            )
            existing.add((tla, source["text_key"]))
            count += 1
            if count >= 20:
                break
    return generated


def summary(metrics):
    out = {}
    for task in ["acceptance", "manual_order", "crum_style"]:
        items = [r for r in metrics if r["task"] == task]
        out[task] = dict(
            groups=len(items),
            ndcg_at_6=(
                float(np.mean([r["ndcg_at_6"] for r in items])) if items else None
            ),
            pair_accuracy=(
                float(np.mean([r["pair_accuracy"] for r in items])) if items else None
            ),
        )
        if task == "acceptance":
            out[task]["known_precision_at_3"] = (
                float(np.mean([r["known_precision_at_3"] for r in items]))
                if items
                else None
            )
    return out


def difference(left, right, task):
    a = {r["group"]: r for r in left if r["task"] == task}
    b = {r["group"]: r for r in right if r["task"] == task}
    clusters = defaultdict(list)
    for key in sorted(a.keys() & b.keys()):
        clusters[a[key]["family"]].append(a[key]["ndcg_at_6"] - b[key]["ndcg_at_6"])
    if not clusters:
        return dict(groups=0, delta_ndcg=None, ci95=None)
    values = list(clusters.values())
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(1000):
        boot.append(
            float(
                np.mean(
                    [
                        v
                        for i in rng.integers(len(values), size=len(values))
                        for v in values[i]
                    ]
                )
            )
        )
    return dict(
        groups=sum(map(len, values)),
        lemma_clusters=len(values),
        delta_ndcg=float(np.mean([v for vs in values for v in vs])),
        ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
    )


def run(config, rebuild=False):
    path = config.derived / "ablation_rows.jsonl"
    stamp = config.derived / "ablation_fingerprint.json"
    current = feature_fingerprint(config)
    rebuild = rebuild or not stamp.exists() or json.loads(stamp.read_text()) != current
    rows = (
        build_rows(config)
        if rebuild or not path.exists()
        else [json.loads(l) for l in path.open()]
    )
    lex = load(config.lexicon)
    vectors = CachedVectors(config)
    all_results = []
    predictions = []
    negative_evidence = []
    setups = [
        ("random", [], "both", "continuous", "none"),
        ("gdex_rule", GDEX, "both", "continuous", "none"),
        (
            "tree_no_sent_len",
            [c for c in CHEAP if c != "sent_len"],
            "both",
            "continuous",
            "none",
        ),
        (
            "tree_no_length",
            [
                c
                for c in CHEAP
                if c
                not in {
                    "sent_len",
                    "n_chars",
                    "avg_word_length",
                    "max_word_length",
                    "min_word_length",
                    "trans_len",
                    "n_tokens",
                }
            ],
            "both",
            "continuous",
            "none",
        ),
        ("tree_gdex", GDEX, "both", "continuous", "none"),
        ("tree_ud", UD, "both", "continuous", "none"),
        ("tree_gdex_ud", GDEX + UD, "both", "continuous", "none"),
        ("tree_full_no_sense", COLUMNS, "both", "continuous", "none"),
        ("tree_full", CHEAP, "both", "continuous", "none"),
        ("tree_tab_only", CHEAP, "tab", "continuous", "none"),
        ("quality_first_joint", CHEAP, "both", "continuous", "none"),
        ("two_head_baseline", COLUMNS, "both", "continuous", "none"),
        (
            "tree_full_no_gdex",
            UD + INFO + ["sense_overlap"],
            "both",
            "continuous",
            "none",
        ),
        (
            "tree_full_no_ud",
            GDEX + INFO + ["sense_overlap"],
            "both",
            "continuous",
            "none",
        ),
        (
            "tree_full_no_info",
            GDEX + UD + ["sense_overlap"],
            "both",
            "continuous",
            "none",
        ),
        ("tree_extended_ud", CHEAP + EXTRA, "both", "continuous", "none"),
        ("tree_crum_only", CHEAP, "crum", "continuous", "none"),
        ("tree_321_floor", CHEAP, "both", "legacy321", "none"),
        ("tree_naive_homograph", CHEAP, "both", "continuous", "naive"),
        ("tree_barley_father", CHEAP, "both", "continuous", "barley_father"),
    ]
    semantic_setups = [
        ("crum4_original", [], "both", "continuous", "none"),
        ("tree_cached_cheap", CHEAP, "both", "continuous", "none"),
        ("tree_cached_bge", SEMANTIC, "both", "continuous", "none"),
        ("tree_cached_bge_barley", SEMANTIC, "both", "continuous", "barley_father"),
    ]
    for dialect in Dialects().data:
        dialect_rows = [r for r in rows if r["dialect"] == dialect]
        if not dialect_rows:
            continue
        families = sorted({r["family"] for r in dialect_rows})
        np.random.default_rng(42).shuffle(families)
        foldmap = {f: i % 5 for i, f in enumerate(families)}
        for scope, configs in [
            ("full_cheap", setups),
            ("cached_BGE_complete_case", semantic_setups),
        ]:
            data = [
                r
                for r in dialect_rows
                if scope == "full_cheap" or r["cosine"] is not None
            ]
            bysetup = defaultdict(list)
            folds = []
            for fold in range(5):
                val = [r for r in data if foldmap[r["family"]] == fold]
                held = {r["text_key"] for r in val}
                train = [
                    r
                    for r in data
                    if foldmap[r["family"]] != fold and r["text_key"] not in held
                ]
                if not val:
                    continue
                check = dict(
                    fold=fold,
                    training_rows=len(train),
                    validation_rows=len(val),
                    purged_training_rows=sum(
                        foldmap[r["family"]] != fold and r["text_key"] in held
                        for r in data
                    ),
                    negative_counts={},
                )
                assert not {r["text_key"] for r in train} & held
                for name, columns, source, scheme, negative in configs:
                    tr = [
                        r
                        for r in train
                        if source == "both"
                        or (r["task"] == "crum_style") == (source == "crum")
                    ]
                    neg = generate_negatives(tr, negative, lex, vectors)
                    if scope != "full_cheap":
                        neg = [r for r in neg if r["cosine"] is not None]
                    assert not any(
                        r["text_key"] in held or foldmap[r["family"]] == fold
                        for r in neg
                    )
                    check["negative_counts"][name] = len(neg)
                    if negative == "barley_father":
                        for r in neg:
                            negative_evidence.append(
                                dict(
                                    scope=scope,
                                    dialect=dialect,
                                    fold=fold,
                                    target=r["TLA"],
                                    source=r["source_tla"],
                                    text_key=r["text_key"],
                                    translation=r["candidate"]["translation"],
                                )
                            )
                    if name == "random":
                        scores = np.random.default_rng(1000 + fold).random(len(val))
                    elif name == "gdex_rule":
                        scores = [r["heuristic"] for r in val]
                    elif name == "crum4_original":
                        maximum = {}
                        for r in val:
                            maximum[r["group"]] = max(
                                maximum.get(r["group"], 0.0),
                                math.log1p(r["old_frequency"]),
                            )
                        scores = [
                            0.55 * r["cosine"]
                            + 0.30 * r["rouge"]
                            + 0.15
                            * math.log1p(r["old_frequency"])
                            / max(maximum[r["group"]], 1e-9)
                            for r in val
                        ]
                    else:
                        model = fit_ranker(tr + neg, columns, scheme)
                        scores = predict(model, val, columns)
                        if name in {"quality_first_joint", "two_head_baseline"}:
                            scores = quality_first(
                                tr, val, scores, name == "two_head_baseline"
                            )
                    ms = evaluate(val, scores)
                    for m in ms:
                        m["fold"] = fold
                    bysetup[name].extend(ms)
                    predictions.extend(
                        dict(
                            dialect=dialect,
                            scope=scope,
                            setup=name,
                            fold=fold,
                            row_id=r["row_id"],
                            score=float(p),
                        )
                        for r, p in zip(val, scores, strict=True)
                    )
                folds.append(check)
                print(dialect, scope, "fold", fold + 1, "complete", flush=True)
            comparisons = []
            pairs = (
                [
                    ("tree_gdex_ud", "tree_gdex"),
                    ("tree_full_no_sense", "tree_gdex_ud"),
                    ("tree_full", "tree_full_no_sense"),
                    ("tree_full", "tree_full_no_gdex"),
                    ("tree_full", "tree_full_no_ud"),
                    ("tree_full", "tree_full_no_info"),
                    ("tree_extended_ud", "tree_full"),
                    ("tree_no_sent_len", "tree_full"),
                    ("tree_no_length", "tree_full"),
                    ("quality_first_joint", "tree_full"),
                    ("quality_first_joint", "two_head_baseline"),
                    ("tree_full", "tree_tab_only"),
                    ("tree_full", "tree_crum_only"),
                    ("tree_full", "tree_321_floor"),
                    ("tree_naive_homograph", "tree_full"),
                    ("tree_barley_father", "tree_full"),
                ]
                if scope == "full_cheap"
                else [
                    ("tree_cached_bge", "tree_cached_cheap"),
                    ("tree_cached_bge", "crum4_original"),
                    ("tree_cached_bge_barley", "tree_cached_bge"),
                ]
            )
            for left, right in pairs:
                comparisons.append(
                    dict(
                        setup=left,
                        reference=right,
                        by_task={
                            task: difference(bysetup[left], bysetup[right], task)
                            for task in ["acceptance", "manual_order", "crum_style"]
                        },
                    )
                )
            all_results.append(
                dict(
                    dialect=dialect,
                    scope=scope,
                    rows=len(data),
                    families=len(families),
                    summaries=[
                        dict(setup=name, features=len(cols), **summary(bysetup[name]))
                        for name, cols, *_ in configs
                    ],
                    comparisons=comparisons,
                    folds=folds,
                    group_metrics=dict(bysetup),
                )
            )
    vectors.connection.close()
    write_json(
        config.reports / "ablation_results.json",
        dict(
            provenance=current,
            protocol=dict(
                folds=5,
                seed=42,
                split="spelling family including homographs; exact Coptic text purged across folds",
                evaluation="same original validation candidates per scope; continuous Crum gain; no synthetic validation labels; expected DCG for score ties",
                labels="TAB priority 1/2/3 -> 1,2/3,1/3; unranked selected only binary acceptance; forbidden 0. Crum observed order -> 1..0.2 within article",
                legacy_labels="TAB unchanged; Crum max(1,3-0.1*(raw_rank-1))/3 floor",
                training="100 rounds, 15 leaves, task/source group weight normalized; Crum weight .35, TAB 1",
                semantic="only cached real BGE complete cases, identical candidates across semantic comparison; no model loaded",
                uncertainty="paired spelling-family cluster bootstrap 1000 resamples; model selection uses same folds, exploratory",
                caveat="known-candidate preference, not whole-corpus precision; mixed sources have different tasks, reported separately",
            ),
            results=all_results,
        ),
    )
    write_json(config.reports / "ablation_predictions.json", predictions)
    write_json(
        config.reports / "barley_negative_training_audit.json", negative_evidence
    )
    return all_results
