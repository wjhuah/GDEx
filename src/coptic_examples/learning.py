"""Training labels and models for inclusion and editorial preference."""

import hashlib, json, math
from collections import Counter, defaultdict
import numpy as np
import lightgbm as lgb
from .dialects import Dialects
from .config import write_json
from .annotations import load_manual, forbidden_rules
from .corpus import Corpus
from .dataset import text_key
from .lexicon import load
from .sense import native_candidates, align_candidate
from .inclusion import EntryFeatures, COLUMNS_INCLUSION
from .text import real_translation, editorial_heading


def crum_grade(position, total):
    return 3.0 if total <= 1 else 3.0 - 1.5 * (position - 1) / (total - 1)


def tab_grade(priority, maximum=3.0):
    if priority is None:
        return None
    return 3.0 - 2.0 * (float(priority) - 1) / max(1.0, maximum - 1)


def build_examples(config):
    raw = [json.loads(s) for s in (config.derived / "ablation_rows.jsonl").open()]
    lex = load(config.lexicon)
    annotations = load_manual(config.manual)
    rules = forbidden_rules(annotations)
    records = {}
    conflicts = []
    for r in raw:
        if not r.get("TLA") or r["task"] not in {"acceptance", "crum_style"}:
            continue
        key = (r["dialect"], r["TLA"], r["text_key"])
        negative = r["task"] == "acceptance" and r["gain"] == 0
        blocked = "ALL" in {s.upper() for s in rules[r["TLA"]]}
        if r["task"] == "crum_style" and blocked:
            conflicts.append(
                dict(
                    tla=r["TLA"],
                    sentence_id=r["candidate"]["id"],
                    reason="TAB_ALL_overrides_Crum_binding",
                )
            )
            continue
        item = records.setdefault(
            key,
            dict(
                tla=r["TLA"],
                family=r["family"],
                dialect=r["dialect"],
                text_key=r["text_key"],
                candidate=r["candidate"],
                y=1,
                ranks=[],
                sources=[],
            ),
        )
        if negative or blocked:
            item["y"] = -1
        if r["task"] == "acceptance":
            item["sources"].append("tab_forbidden" if negative else "tab")
            if not negative and r.get("priority") is not None:
                item["ranks"].append(
                    dict(
                        source="tab",
                        grade=tab_grade(r["priority"]),
                        priority=r["priority"],
                    )
                )
        else:
            item["sources"].append("crum")
            item.setdefault("crum_refs", []).append(
                dict(
                    occurrence_id=r.get("occurrence_id"),
                    group=r["group"],
                    order=r["rank"],
                )
            )
    # Expand ALL rules over the complete attested pool instead of a training cap.
    corpus = Corpus(config)
    ledger = []
    for tla, rr in rules.items():
        if "ALL" not in {s.upper() for s in rr} or tla not in lex:
            continue
        import copy

        entry = copy.deepcopy(lex[tla])
        entry.forms = tuple(
            set(entry.forms) | {a["lemma"] for a in annotations if a["TLA"] == tla}
        )
        for dialect in Dialects().data:
            for c in native_candidates(config, corpus, entry, dialect, True):
                ledger.append(
                    dict(
                        tla=tla,
                        dialect=dialect,
                        sentence_id=c["id"],
                        label=-1,
                        source="TAB_ALL",
                    )
                )
                if not real_translation(c["translation"]) or editorial_heading(
                    c["translation"]
                ):
                    continue
                key = (dialect, tla, text_key(c))
                item = records.setdefault(
                    key,
                    dict(
                        tla=tla,
                        family=entry.lemma,
                        dialect=dialect,
                        text_key=key[-1],
                        candidate=c,
                        y=-1,
                        ranks=[],
                        sources=["tab_forbidden"],
                    ),
                )
                item["y"] = -1
                item["all_forbidden"] = True
                item["ranks"] = []
                item["sources"] = list(set(item["sources"]) | {"tab_forbidden"})
    # Crum order is kept separately from its positive inclusion label.
    from .config import database

    refs = {}
    with database(config.work / "cache/crum_bible.sqlite3") as db:
        for oid, payload in db.execute("SELECT id,payload FROM occurrence"):
            p = json.loads(payload)
            page = p["page"]
            digits = "".join(c for c in page if c.isdigit())
            refs[oid] = (
                int(digits or 0),
                int(page.endswith("b")),
                p["file"],
                p["char_start"],
            )
    grouped = defaultdict(list)
    for item in records.values():
        if item["y"] > 0 and item.get("crum_refs"):
            grouped[(item["dialect"], item["tla"])].append(item)
    for values in grouped.values():
        values.sort(
            key=lambda r: min(
                refs.get(c["occurrence_id"], (0, 0, c["group"], c["order"]))
                for c in r["crum_refs"]
            )
        )
        for i, item in enumerate(values, 1):
            item["ranks"].append(
                dict(
                    source="crum",
                    grade=crum_grade(i, len(values)),
                    position=i,
                    total=len(values),
                )
            )
    result = list(records.values())
    for r in result:
        if r["y"] < 0:
            r["ranks"] = []
        r["sources"] = sorted(set(r["sources"]))
    write_json(
        config.reports / "learning/negative_rules.json",
        dict(
            ledger=ledger,
            counts=dict(Counter(r["tla"] for r in ledger)),
            alignment_conflicts=conflicts,
        ),
    )
    config.derived.mkdir(exist_ok=True, parents=True)
    with (config.derived / "learning_examples.jsonl").open("w") as f:
        for r in result:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return result


def fit_inclusion(rows):
    y = np.asarray([r["y"] > 0 for r in rows], dtype=int)
    counts = np.bincount(y, minlength=2)
    if not all(counts):
        return None
    weights = np.asarray([len(rows) / (2 * counts[v]) for v in y])
    d = lgb.Dataset(
        np.asarray([r["x"] for r in rows]),
        label=y,
        weight=weights,
        feature_name=COLUMNS_INCLUSION,
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
        d,
        num_boost_round=120,
    )


def rank_observations(rows):
    out = []
    for r in rows:
        if r["y"] < 0:
            out.append(dict(r, target=-1.0, bounded=False, rank_source="negative"))
        elif r["ranks"]:
            out.extend(
                dict(r, target=p["grade"], bounded=False, rank_source=p["source"])
                for p in r["ranks"]
            )
        else:
            out.append(
                dict(r, target=1.0, bounded=True, rank_source="unranked_positive")
            )
    return out


def fit_score(rows, joint=False):
    values = rank_observations(rows)
    if not joint:
        values = [r for r in values if r["y"] > 0 and not r["bounded"]]
    if not values:
        return None
    target = np.asarray([r["target"] for r in values])
    bounded = np.asarray([r["bounded"] for r in values])
    weights = np.ones(len(values))
    if joint:
        positive = np.asarray([r["y"] > 0 for r in values])
        counts = Counter(positive)
        weights = np.asarray([len(values) / (2 * counts[v]) for v in positive])

    def objective(pred, data):
        delta = pred - target
        active = ~bounded | (pred < 1.0)
        return delta * active, np.where(active, 1.0, 0.001)

    ds = lgb.Dataset(
        np.asarray([r["x"] for r in values]),
        label=target,
        weight=weights,
        feature_name=COLUMNS_INCLUSION,
    )

    # With a custom objective LightGBM leaves sample-weight application to the callback.
    def weighted_objective(pred, data):
        g, h = objective(pred, data)
        return g * weights, h * weights

    return lgb.train(
        dict(
            objective=weighted_objective if joint else "regression",
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


def predict(model, rows):
    return (
        model.predict(np.asarray([r["x"] for r in rows]), num_threads=4)
        if model
        else np.zeros(len(rows))
    )


def augment(train, ratio, kind, ef, lex, memo):
    if ratio == 0:
        return []
    positives = [r for r in train if r["y"] > 0]
    by = defaultdict(list)
    for r in positives:
        by[r["tla"]].append(r)
    known = {(r["tla"], r["text_key"]) for r in train}
    out = []
    ordered_pool = sorted(
        positives, key=lambda r: hashlib.sha256(r["text_key"].encode()).hexdigest()
    )
    for tla, own in sorted(by.items()):
        wanted = math.ceil(len(own) * ratio)
        hard_n = (
            wanted
            if kind == "hard"
            else 0 if kind == "unrelated" else math.ceil(wanted / 2)
        )
        rivals = {e.tla for e in ef.rivals(lex[tla])}
        hard = [
            r
            for t in sorted(rivals)
            for r in by.get(t, [])
            if (tla, r["text_key"]) not in known
        ]
        hard.sort(
            key=lambda r: (
                hashlib.sha256((tla + r["text_key"]).encode()).hexdigest(),
                r["tla"],
                r["candidate"]["id"],
            )
        )
        chosen = []
        seen = set()
        for source in hard:
            if source["text_key"] in seen:
                continue
            import copy, re

            probe = copy.deepcopy(source["candidate"])
            target_ids = {
                probe["tokens"][i]["id"] for i in align_candidate(probe, lex[tla], True)
            }
            source_ids = {
                probe["tokens"][i]["id"]
                for i in align_candidate(probe, lex[source["tla"]], True)
            }
            if not target_ids.intersection(source_ids):
                continue
            # Co-occurring explicit meanings give an ambiguous sentence-level negative.
            own_words = set(re.findall(r"[a-z]+", lex[tla].gloss.lower()))
            rival_words = set(re.findall(r"[a-z]+", lex[source["tla"]].gloss.lower()))
            stop = {
                "a",
                "an",
                "the",
                "to",
                "of",
                "be",
                "in",
                "and",
                "or",
                "with",
                "for",
                "as",
                "by",
                "on",
                "from",
                "sb",
                "sth",
            }
            translated = set(re.findall(r"[a-z]+", probe["translation"].lower()))
            if (own_words - rival_words - stop) & translated and (
                rival_words - own_words - stop
            ) & translated:
                continue
            chosen.append((source, "homograph"))
            seen.add(source["text_key"])
            if len(chosen) >= hard_n:
                break
        if hard_n == 0:
            chosen = []
            seen = set()
        unrelated_n = wanted - len(chosen) if kind != "hard" else 0
        if unrelated_n:
            start = int(hashlib.sha256(tla.encode()).hexdigest()[:8], 16) % max(
                1, len(ordered_pool)
            )
            candidates = (
                ordered_pool[(start + j) % len(ordered_pool)]
                for j in range(len(ordered_pool))
            )
            for source in candidates:
                if source["text_key"] in seen or (tla, source["text_key"]) in known:
                    continue
                import copy

                c = copy.deepcopy(source["candidate"])
                if align_candidate(c, lex[tla], True):
                    continue
                chosen.append((source, "unrelated"))
                seen.add(source["text_key"])
                unrelated_n -= 1
                if unrelated_n == 0:
                    break
        for source, reason in chosen:
            key = (tla, source["text_key"])
            if key in known:
                continue
            feature_key = (tla, source["candidate"]["id"])
            if feature_key not in memo:
                import copy

                c = copy.deepcopy(source["candidate"])
                align_candidate(c, lex[tla], True)
                memo[feature_key] = ef.vector(c, lex[tla])
            out.append(
                dict(
                    source,
                    tla=tla,
                    family=own[0]["family"],
                    x=memo[feature_key],
                    y=-1,
                    ranks=[],
                    sources=[reason],
                    synthetic=True,
                    source_tla=source["tla"],
                )
            )
            known.add(key)
    return out


class LearningModel:
    def __init__(self, config, lex, dialect):
        self.config = config
        self.dialect = dialect
        self.folder = config.work / "models" / dialect
        self.metadata = json.loads((self.folder / "learning_v2.json").read_text())
        if (
            self.metadata["snapshot_sha256"]
            != hashlib.sha256(
                (config.work / "cache/cdo_index.json").read_bytes()
            ).hexdigest()
        ):
            raise RuntimeError("Retrain after updating the dictionary snapshot")
        self.features = EntryFeatures(config, lex, dialect)
        self.models = {
            name: lgb.Booster(model_file=str(self.folder / (name + ".txt")))
            for name in ["inclusion_v2", "ranking_v2", "joint_v2"]
            if (self.folder / (name + ".txt")).exists()
        }

    def score(self, candidates, entry, architecture):
        matrix = np.asarray([self.features.vector(c, entry) for c in candidates])
        if architecture == "joint":
            raw = self.models["joint_v2"].predict(matrix, num_threads=4)
            return 1 / (1 + np.exp(-np.clip(raw, -30, 30))), raw
        inc = (
            self.models["inclusion_v2"].predict(matrix, num_threads=4)
            if "inclusion_v2" in self.models
            else np.full(len(candidates), 0.5)
        )
        rank = (
            self.models["ranking_v2"].predict(matrix, num_threads=4)
            if "ranking_v2" in self.models
            else np.full(len(candidates), 2.0)
        )
        return inc, rank

    def close(self):
        self.features.close()


def deduplicate_candidates(rows):
    """Retain all source selections when identical texts occur in several corpora."""
    unique = {}
    for row in rows:
        key = text_key(row)
        if key not in unique:
            unique[key] = row
            continue
        kept = unique[key]
        for citation in row.get("crum_citations", []):
            if citation not in kept.setdefault("crum_citations", []):
                kept["crum_citations"].append(citation)
        kept.setdefault("alternate_ids", []).append(row["id"])
        kept["manual_selected"] = kept.get("manual_selected", False) or row.get(
            "manual_selected", False
        )
        priorities = [
            v
            for v in [kept.get("manual_priority"), row.get("manual_priority")]
            if v is not None
        ]
        kept["manual_priority"] = min(priorities) if priorities else None
    return list(unique.values()), len(rows) - len(unique)


def recommend(config, lex, entry, dialect, k, architecture):
    from .relevance import DefinitionGate
    from .annotations import forbidden, resolve_manual
    from .shortlist import numbers
    from .editorial import order

    corpus = Corpus(config)
    annotations = load_manual(config.manual)
    rules = forbidden_rules(annotations)[entry.tla]
    candidates = native_candidates(config, corpus, entry, dialect, True)
    excluded = Counter()
    out = []
    if "ALL" in {r.upper() for r in rules}:
        excluded["TAB_ALL"] = len(candidates)
        return dict(
            TLA=entry.tla,
            lemma=entry.lemma,
            definition=entry.gloss,
            dialect=dialect,
            candidate_count=len(candidates),
            eligible_count=0,
            shortlist=[],
            ranker="unified" if architecture == "joint" else "research",
            model_status={"provisional": False},
            warnings=[],
            excluded=dict(excluded),
            definition_gate_applied=False,
            architecture=architecture,
        )
    manual, _ = resolve_manual(
        [a for a in annotations if a["TLA"] == entry.tla and a["URN"].upper() != "X"],
        corpus,
    )
    priorities = {
        r["candidate"]["id"]: (
            int(r["annotation"]["priority"])
            if r["annotation"]["priority"].isdigit()
            else None
        )
        for r in manual
        if r["candidate"]["dialect_id"] == dialect
    }
    evidence = DefinitionGate(config, lex, entry, dialect)
    for c in candidates:
        if forbidden(c, rules, c["id"] in priorities):
            excluded["TAB_forbidden"] += 1
            continue
        if not real_translation(c["translation"]) or editorial_heading(
            c["translation"]
        ):
            excluded["untranslated_or_heading"] += 1
            continue
        c["crum_citations"] = [
            r for r in evidence.citations[c["id"]] if r["tla"] == entry.tla
        ]
        c.update(
            tla=entry.tla,
            manual_selected=c["id"] in priorities,
            manual_priority=priorities.get(c["id"]),
        )
        out.append(c)
    evidence.close()
    out, duplicates = deduplicate_candidates(out)
    excluded["duplicate"] = duplicates
    if out:
        model = LearningModel(config, lex, dialect)
        inc, ranking = model.score(out, entry, architecture)
        for c, p, s in zip(out, inc, ranking):
            c.update(
                learned_inclusion_score=float(p),
                shortlist_score=float(s),
                score=float(s),
                acceptance_score=float(p),
                quality_score=float(p),
            )
        threshold = model.metadata.get("thresholds", {}).get(architecture, 0.5)
        eligible = [
            c
            for c in out
            if c["manual_selected"]
            or c["crum_citations"]
            or (c["learned_inclusion_score"] > threshold)
        ]
        excluded["model_rejected"] = len(out) - len(eligible)
        selected = order(eligible)[:k]
        for c in selected:
            from .text import sm_freq, rouge
            from .relevance import bleu2

            c["definition_relevance"] = dict(
                freq_match=sm_freq(entry.gloss, c["translation"]),
                cosine=model.features.vectors.cosine(entry.gloss, c["translation"]),
                rouge_l=rouge(entry.gloss, c["translation"]),
                bleu2=bleu2(entry.gloss, c["translation"]),
            )
            _, syntax, _ = numbers(c, entry, model.features.extractor)
            c["syntax"] = syntax
            c["information_score"] = syntax["features"]["information_score"]
        model.close()
    else:
        threshold = model.metadata.get("thresholds", {}).get(architecture, 0.5)
        eligible = []
        selected = []
    return dict(
        TLA=entry.tla,
        lemma=entry.lemma,
        definition=entry.gloss,
        dialect=dialect,
        candidate_count=len(candidates),
        eligible_count=len(eligible),
        shortlist=selected,
        ranker="unified" if architecture == "joint" else "research",
        model_status={"provisional": dialect != "sahidic"},
        warnings=[],
        excluded=dict(excluded),
        definition_gate_applied=False,
        architecture=architecture,
    )
