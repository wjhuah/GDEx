"""Dialect-specific inclusion and editorial preference models."""

import hashlib
import json
import math
from collections import Counter, defaultdict
import numpy as np
from .config import write_json, database
from .corpus import Corpus
from .dataset import read_rows, text_key
from .dialects import Dialects
from .features import Features, GDEX, UD, quality
from .lexicon import load
from .selection import profile, choose, metrics
from .text import real_translation, editorial_heading, sm_freq, norm, lemma_key

INFO = [
    "context_content_count",
    "target_descriptor_count",
    "filled_argument_roles",
    "target_occurrences",
    "context_content_diversity",
    "proper_name_ratio",
    "information_score",
    "dependency_target_found",
]
COLUMNS = GDEX + UD + INFO


def numbers(candidate, entry, extractor):
    from .sense import targets

    indices = targets(candidate, entry)
    syntax = profile(candidate, entry.lemma, entry.forms, allowed_indices=indices)
    f = extractor.extract(candidate, entry.lemma, target_indices=indices)
    f.update(syntax["features"])
    return [float(f[k]) for k in COLUMNS], syntax, quality(f)


def fit_linear(x, y, intercept=True):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    scaler = StandardScaler(with_mean=intercept).fit(x)
    model = LogisticRegression(
        C=0.25,
        class_weight="balanced",
        fit_intercept=intercept,
        solver="liblinear" if intercept else "lbfgs",
        max_iter=2000,
        random_state=42,
    ).fit(scaler.transform(x), y)
    return dict(
        columns=COLUMNS,
        mean=scaler.mean_.tolist() if intercept else [0.0] * len(scaler.scale_),
        scale=scaler.scale_.tolist(),
        coefficients=model.coef_[0].tolist(),
        intercept=float(model.intercept_[0]) if intercept else 0.0,
    )


def decision(model, x):
    x = np.asarray(x, dtype=float)
    if not len(x):
        return np.asarray([])
    if model.get("kind") == "lightgbm":
        import lightgbm as lgb

        predictor = model.get("predictor") or lgb.Booster(
            model_file=model["model_file"]
        )
        return predictor.predict(x, raw_score=True, num_threads=4)
    return ((x - np.asarray(model["mean"])) / np.asarray(model["scale"])) @ np.asarray(
        model["coefficients"]
    ) + model["intercept"]


def tree_acceptance(rows):
    import lightgbm as lgb

    y = np.asarray([r["y"] for r in rows])
    counts = Counter(y)
    weights = np.asarray([len(y) / (2 * counts[v]) for v in y])
    data = lgb.Dataset(
        np.asarray([r["x"] for r in rows]),
        label=y,
        weight=weights,
        feature_name=COLUMNS,
    )
    model = lgb.train(
        dict(
            objective="binary",
            learning_rate=0.05,
            num_leaves=15,
            min_data_in_leaf=15,
            lambda_l2=1.0,
            verbosity=-1,
            num_threads=4,
            seed=42,
            deterministic=True,
            force_col_wise=True,
        ),
        data,
        num_boost_round=120,
    )
    return dict(kind="lightgbm", columns=COLUMNS, predictor=model)


def tree_style(rows):
    import lightgbm as lgb

    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    ordered = []
    labels = []
    sizes = []
    maximum = 0
    for group, items in sorted(groups.items()):
        levels = sorted({r["rank"] for r in items})
        if len(levels) < 2:
            continue
        grade = {rank: len(levels) - 1 - i for i, rank in enumerate(levels)}
        ordered.extend(items)
        labels.extend(grade[r["rank"]] for r in items)
        sizes.append(len(items))
        maximum = max(maximum, len(levels) - 1)
    if not ordered:
        return None
    data = lgb.Dataset(
        np.asarray([r["x"] for r in ordered]),
        label=labels,
        group=sizes,
        feature_name=COLUMNS,
    )
    model = lgb.train(
        dict(
            objective="lambdarank",
            label_gain=list(range(maximum + 1)),
            learning_rate=0.05,
            num_leaves=15,
            min_data_in_leaf=10,
            lambda_l2=1.0,
            verbosity=-1,
            num_threads=4,
            seed=42,
            deterministic=True,
            force_col_wise=True,
        ),
        data,
        num_boost_round=120,
    )
    return dict(kind="lightgbm", columns=COLUMNS, predictor=model)


def style_accuracy(rows, scores, cap=48):
    groups = defaultdict(list)
    wins = []
    for r, score in zip(rows, scores, strict=True):
        groups[r["group"]].append((r, float(score)))
    for group, items in sorted(groups.items()):
        items.sort(key=lambda x: (x[0]["rank"], x[0]["text_key"]))
        pairs = [
            (i, j)
            for i in range(len(items))
            for j in range(i + 1, len(items))
            if items[i][0]["rank"] != items[j][0]["rank"]
        ]
        pairs.sort(
            key=lambda ij: hashlib.sha256((group + str(ij)).encode()).hexdigest()
        )
        for i, j in pairs[:cap]:
            wins.append(
                1.0
                if items[i][1] > items[j][1]
                else 0.5 if items[i][1] == items[j][1] else 0.0
            )
    return (float(np.mean(wins)), len(wins)) if wins else (None, 0)


def save_predictor(path, model):
    if model.get("kind") == "lightgbm":
        target = path.with_suffix(".txt")
        model["predictor"].save_model(str(target))
        write_json(path, dict(kind="lightgbm", columns=COLUMNS, model_file=target.name))
    else:
        write_json(path, model)


def pair_data(rows, cap=48):
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    deltas = []
    labels = []
    for group, items in sorted(groups.items()):
        items = sorted(items, key=lambda r: (r["rank"], r["text_key"]))
        pairs = [
            (i, j)
            for i in range(len(items))
            for j in range(i + 1, len(items))
            if items[i]["rank"] != items[j]["rank"]
        ]
        pairs.sort(
            key=lambda ij: hashlib.sha256((group + str(ij)).encode()).hexdigest()
        )
        for i, j in pairs[:cap]:
            diff = np.asarray(items[i]["x"]) - np.asarray(items[j]["x"])
            deltas.extend([diff.tolist(), (-diff).tolist()])
            labels.extend([1, 0])
    return np.asarray(deltas), np.asarray(labels)


def train(config):
    corpus = Corpus(config)
    registry = Dialects()
    lex = load(config.lexicon)
    frequencies = defaultdict(Counter)
    for c in corpus.all():
        frequencies[c["dialect_id"]].update(t["form"] for t in c["tokens"])
    evidence = read_rows(config)
    summary = []
    audit = config.work / "cache/crum_bible.sqlite3"
    for dialect in registry.data:
        directory = config.work / "models" / dialect
        directory.mkdir(parents=True, exist_ok=True)
        freq = dict(frequencies[dialect])
        extractor = Features(freq)
        write_json(directory / "frequencies.json", freq)
        accepted = []
        seen = set()
        for r in evidence:
            if (
                r.get("dialect_id") != dialect
                or r["TLA"] not in lex
                or r["evidence"] not in {"manual", "manual_forbidden_rule"}
            ):
                continue
            key = (r["TLA"], r["text_key"])
            if key in seen:
                continue
            seen.add(key)
            entry = lex[r["TLA"]]
            c = r["candidate"]
            if not real_translation(c["translation"]) or editorial_heading(
                c["translation"]
            ):
                continue
            x, _, _ = numbers(c, entry, extractor)
            accepted.append(
                dict(
                    x=x,
                    y=int(r["evidence"] == "manual"),
                    lemma=entry.lemma,
                    text_key=r["text_key"],
                    group=r["TLA"],
                )
            )
        style = []
        style_seen = set()
        with database(audit) as db, corpus.connect() as source:
            query = "SELECT o.payload,l.sentence_id FROM occurrence o JOIN link l ON o.id=l.occurrence_id WHERE l.dialect=? AND l.status='verified_target_form' ORDER BY o.file,o.article,o.editorial_order,l.sentence_id"
            for payload, sid in db.execute(query, (dialect,)):
                r = json.loads(payload)
                c = json.loads(
                    source.execute(
                        "SELECT payload FROM sentence WHERE id=?", (sid,)
                    ).fetchone()[0]
                )
                key = (r["article_id"], text_key(c))
                possible = []
                for tla in r["tla_candidates"]:
                    e = lex[tla]
                    codes = {
                        code for v in (e.form_dialects or {}).values() for code in v
                    }
                    if codes and dialect not in registry.from_codes(codes):
                        continue
                    if profile(c, e.lemma, e.forms)["target_indices"]:
                        possible.append(tla)
                if not possible:
                    continue
                # Identical lemma/POS candidates produce identical local features.
                # This learns an article's ordering without pretending to settle
                # which of the synonymous TLA IDs its historical heading means.
                if len({(lex[t].lemma, lex[t].pos) for t in possible}) != 1:
                    continue
                tla = possible[0]
                entry = lex[tla]
                if (
                    key in style_seen
                    or not real_translation(c["translation"])
                    or editorial_heading(c["translation"])
                ):
                    continue
                x, syn, _ = numbers(c, entry, extractor)
                if not syn["target_indices"]:
                    continue
                style_seen.add(key)
                style.append(
                    dict(
                        x=x,
                        rank=r["editorial_order"],
                        group=r["article_id"],
                        lemma=entry.lemma,
                        text_key=text_key(c),
                        TLA=tla if len(possible) == 1 else None,
                        tla_candidates=possible,
                        occurrence_id=r["id"],
                        sentence_id=sid,
                    )
                )
        # Two tasks, two model files; no manufactured 3/2/1 label from a Crum position.
        report = dict(
            dialect=dialect,
            acceptance_rows=len(accepted),
            acceptance_positive=sum(r["y"] for r in accepted),
            style_rows=len(style),
            style_groups=len({r["group"] for r in style}),
            features=len(COLUMNS),
        )
        if len({r["y"] for r in accepted}) == 2:
            model = fit_linear([r["x"] for r in accepted], [r["y"] for r in accepted])
            write_json(directory / "acceptance.json", model)
            report["acceptance_status"] = "trained"
        else:
            (directory / "acceptance.json").unlink(missing_ok=True)
            report["acceptance_status"] = (
                "insufficient_two_class_labels; explicit heuristic fallback"
            )
        x, y = pair_data(style)
        if len(y):
            write_json(directory / "crum_style.json", fit_linear(x, y, False))
            report["style_status"] = "trained"
            report["training_pairs"] = len(y) // 2
        else:
            (directory / "crum_style.json").unlink(missing_ok=True)
            report["style_status"] = "insufficient_ordered_pairs"
        # Auditable fixed lemma split; purge repeated Coptic spans across folds.
        families = sorted({r["lemma"] for r in accepted + style})
        np.random.default_rng(42).shuffle(families)
        foldmap = {f: i % 5 for i, f in enumerate(families)}
        cv = []
        for fold in range(5):
            aq = [r for r in accepted if foldmap[r["lemma"]] == fold]
            sq = [r for r in style if foldmap[r["lemma"]] == fold]
            held = {r["text_key"] for r in aq + sq}
            at = [
                r
                for r in accepted
                if foldmap[r["lemma"]] != fold and r["text_key"] not in held
            ]
            st = [
                r
                for r in style
                if foldmap[r["lemma"]] != fold and r["text_key"] not in held
            ]
            check = dict(
                fold=fold,
                acceptance_validation=len(aq),
                style_validation=len(sq),
                held_out_unique_texts=len(held),
            )
            if len({r["y"] for r in at}) == 2 and len({r["y"] for r in aq}) == 2:
                from sklearn.metrics import roc_auc_score, average_precision_score

                m = fit_linear([r["x"] for r in at], [r["y"] for r in at])
                scores = decision(m, [r["x"] for r in aq])
                check.update(
                    acceptance_auc=float(roc_auc_score([r["y"] for r in aq], scores)),
                    acceptance_average_precision=float(
                        average_precision_score([r["y"] for r in aq], scores)
                    ),
                )
                tree = tree_acceptance(at)
                pred = decision(tree, [r["x"] for r in aq])
                check["tree_acceptance_auc"] = float(
                    roc_auc_score([r["y"] for r in aq], pred)
                )
            tx, ty = pair_data(st)
            vx, vy = pair_data(sq)
            if len(ty) and len(vy):
                m = fit_linear(tx, ty, False)
                pred = decision(m, vx)
                check.update(
                    style_pair_accuracy=float(np.mean((pred > 0) == vy)),
                    style_validation_pairs=len(vy) // 2,
                )
                tree = tree_style(st)
                if tree:
                    acc, n = style_accuracy(sq, decision(tree, [r["x"] for r in sq]))
                    check.update(
                        tree_style_pair_accuracy=acc, tree_style_validation_pairs=n
                    )
            cv.append(check)
        report["validation"] = cv
        report["acceptance_mean_auc"] = (
            float(np.mean([r["acceptance_auc"] for r in cv if "acceptance_auc" in r]))
            if any("acceptance_auc" in r for r in cv)
            else None
        )
        report["style_mean_pair_accuracy"] = (
            float(
                np.mean(
                    [r["style_pair_accuracy"] for r in cv if "style_pair_accuracy" in r]
                )
            )
            if any("style_pair_accuracy" in r for r in cv)
            else None
        )
        tree_auc = [r["tree_acceptance_auc"] for r in cv if "tree_acceptance_auc" in r]
        tree_acc = [
            r["tree_style_pair_accuracy"]
            for r in cv
            if r.get("tree_style_pair_accuracy") is not None
        ]
        report["tree_acceptance_mean_auc"] = (
            float(np.mean(tree_auc)) if tree_auc else None
        )
        report["tree_style_mean_pair_accuracy"] = (
            float(np.mean(tree_acc)) if tree_acc else None
        )
        report["selected_acceptance_model"] = "linear"
        report["selected_style_model"] = "linear"
        if (
            tree_auc
            and report["tree_acceptance_mean_auc"] > report["acceptance_mean_auc"]
        ):
            save_predictor(directory / "acceptance.json", tree_acceptance(accepted))
            report["selected_acceptance_model"] = "lightgbm"
        if (
            tree_acc
            and report["tree_style_mean_pair_accuracy"]
            > report["style_mean_pair_accuracy"]
        ):
            save_predictor(directory / "crum_style.json", tree_style(style))
            report["selected_style_model"] = "lightgbm"
        report["validation_scope"] = (
            "exploratory model selection on the same five fixed folds; no independent post-selection test"
        )
        report["provisional"] = (
            len(accepted) < 100 or len(style) < 100 or len(tree_auc) < 3
        )
        with corpus.connect() as db:
            stamp = db.execute(
                "SELECT value FROM metadata WHERE key='fingerprint'"
            ).fetchone()[0]
        report["corpus_fingerprint"] = stamp
        report["dialect_registry_fingerprint"] = registry.fingerprint
        write_json(directory / "shortlist_metadata.json", report)
        summary.append(report)
        print(dialect, report["acceptance_status"], report["style_status"], flush=True)
    write_json(config.reports / "cheap_models_by_dialect.json", summary)
    return summary


class Shortlist:
    def __init__(self, config, dialect="sahidic"):
        self.config = config
        self.dialect = Dialects().require(dialect)
        self.corpus = Corpus(config)
        self.lex = load(config.lexicon)
        directory = config.work / "models" / self.dialect
        self.meta = json.loads((directory / "shortlist_metadata.json").read_text())
        with self.corpus.connect() as db:
            stamp = db.execute(
                "SELECT value FROM metadata WHERE key='fingerprint'"
            ).fetchone()[0]
        if (
            stamp != self.meta["corpus_fingerprint"]
            or Dialects().fingerprint != self.meta["dialect_registry_fingerprint"]
        ):
            raise RuntimeError(
                "Rebuild dialect models after changing corpus or registry"
            )
        self.extractor = Features(
            json.loads((directory / "frequencies.json").read_text())
        )
        self.acceptance = (
            json.loads((directory / "acceptance.json").read_text())
            if (directory / "acceptance.json").exists()
            else None
        )
        self.style = (
            json.loads((directory / "crum_style.json").read_text())
            if (directory / "crum_style.json").exists()
            else None
        )
        for model in [self.acceptance, self.style]:
            if model and model.get("kind") == "lightgbm":
                model["model_file"] = str(directory / model["model_file"])
        from .annotations import load_manual, forbidden_rules

        self.rules = forbidden_rules(load_manual(config.manual))
        self.manual_ids = defaultdict(set)
        self.manual_priorities = defaultdict(dict)
        for r in read_rows(config):
            if r["evidence"] == "manual" and r.get("dialect_id") == self.dialect:
                self.manual_ids[r["TLA"]].add(r["candidate"]["id"])
                self.manual_priorities[r["TLA"]][r["candidate"]["id"]] = r.get(
                    "priority"
                )

    def search(self, tla, k=10, ranker="research"):
        if not 1 <= k <= 30:
            raise ValueError("shortlist size must be in 1..30")
        entry = self.lex[tla.upper()]
        rows = []
        seen = set()
        matrix = []
        excluded = Counter()
        from .annotations import forbidden

        rules = self.rules[entry.tla]
        manual_ids = self.manual_ids[entry.tla]
        from .sense import native_candidates, NATIVE

        candidates = native_candidates(
            self.config,
            self.corpus,
            entry,
            self.dialect,
            allow_pos_mismatch=ranker == "research",
        )
        from .relevance import DefinitionGate

        gate = DefinitionGate(self.config, self.lex, entry, self.dialect)
        definition_review = []
        for c in candidates:
            if c["dialect_id"] != self.dialect:
                raise AssertionError("Cross-dialect candidate leak")
            crum_refs = [
                ref for ref in gate.citations[c["id"]] if ref["tla"] == entry.tla
            ]
            if forbidden(c, rules, c["id"] in manual_ids) and not crum_refs:
                excluded["forbidden"] += 1
                continue
            if not real_translation(c["translation"]) or editorial_heading(
                c["translation"]
            ):
                excluded["unusable_translation_or_heading"] += 1
                continue
            key = text_key(c)
            if key in seen:
                excluded["duplicate_coptic_span"] += 1
                continue
            seen.add(key)
            relevance = gate.check(c, manual_confirmed=c["id"] in manual_ids)
            c["definition_relevance"] = relevance
            relevance["used_as_filter"] = ranker != "research"
            if relevance["status"] != "pass" and ranker != "research":
                excluded["definition_" + relevance["reason"]] += 1
                definition_review.append(
                    dict(
                        id=c["id"],
                        sent=c["sent"],
                        translation=c["translation"],
                        tokens=c["tokens"],
                        target_token_evidence=c["target_token_evidence"],
                        definition_relevance=relevance,
                    )
                )
                continue
            own = sm_freq(entry.gloss, c["translation"])
            other = {"C998": "C992", "C992": "C998"}.get(entry.tla)
            cross = sm_freq(self.lex[other].gloss, c["translation"]) if other else 0.0
            if other and ranker != "research":
                import re

                words = set(re.findall(r"[a-z]+", c["translation"].lower()))
                father = bool(
                    words
                    & {
                        "father",
                        "fathers",
                        "parent",
                        "parents",
                        "ancestor",
                        "ancestors",
                    }
                )
                barley = "barley" in words
                own_supported = barley if entry.tla == "C998" else father
                cross_supported = father if entry.tla == "C998" else barley
                if not own_supported or cross_supported:
                    excluded["unconfirmed_other_sense_or_ambiguous"] += 1
                    continue
            x, syn, q = numbers(c, entry, self.extractor)
            if not syn["target_indices"]:
                excluded["target_syntax_unavailable"] += 1
                continue
            row = dict(
                c,
                tla=entry.tla,
                crum_citations=crum_refs,
                manual_selected=c["id"] in manual_ids,
                manual_priority=self.manual_priorities[entry.tla].get(c["id"]),
                syntax=syn,
                information_score=syn["features"]["information_score"],
                gdex_heuristic=q,
                lexical_sense_support=own,
                sense_status=(
                    "lexical_support" if own else "not_confirmed_by_cheap_features"
                ),
            )
            rows.append(row)
            matrix.append(x)
        if rows:
            a = (
                1 / (1 + np.exp(-np.clip(decision(self.acceptance, matrix), -30, 30)))
                if self.acceptance
                else np.asarray([r["gdex_heuristic"] for r in rows])
            )
            raw = decision(self.style, matrix) if self.style else np.zeros(len(rows))
            order = (
                np.argsort(np.argsort(raw, kind="stable"), kind="stable")
                / max(1, len(rows) - 1)
                if self.style
                else np.full(len(rows), 0.5)
            )
            for row, accept, style in zip(rows, a, order, strict=True):
                row.update(
                    acceptance_score=float(accept), crum_style_percentile=float(style)
                )
                # Public initial tradeoff, not tuned against the diagnostic examples.
                row["shortlist_score"] = (
                    0.60 * float(accept)
                    + 0.20 * float(style)
                    + 0.15 * row["information_score"]
                    + 0.05 * row["lexical_sense_support"]
                )
        gate.close()
        research_used = False
        if ranker == "quality_first" and rows:
            directory = self.config.work / "models" / self.dialect
            research = json.loads((directory / "research_ranker.json").read_text())
            if research.get("file"):
                if (
                    research["corpus_fingerprint"] != self.meta["corpus_fingerprint"]
                    or research["manual_sha256"]
                    != hashlib.sha256(self.config.manual.read_bytes()).hexdigest()
                ):
                    raise RuntimeError(
                        "Rebuild research model after changing corpus or annotations"
                    )
                import lightgbm as lgb

                predictor = lgb.Booster(model_file=str(directory / research["file"]))
                values = [
                    x + [r["lexical_sense_support"]]
                    for x, r in zip(matrix, rows, strict=True)
                ]
                raw = predictor.predict(np.asarray(values), num_threads=4)
                for row, value in zip(rows, raw, strict=True):
                    row["research_raw_score"] = float(value)
                    row["research_order_score"] = float(
                        1 / (1 + math.exp(-max(-30.0, min(30.0, value))))
                    )
                    row["shortlist_score"] = (
                        (
                            0.60 * row["acceptance_score"]
                            + 0.25 * row["research_order_score"]
                            + 0.15 * row["information_score"]
                        )
                        if ranker == "quality_first"
                        else row["research_order_score"]
                    )
                research_used = True
        if ranker == "definition_syntax" and rows:
            syntax_model = json.loads(
                (
                    self.config.work / "models" / self.dialect / "syntax_order.json"
                ).read_text()
            )
            syntax_values = [
                [x[COLUMNS.index(k)] for k in syntax_model["columns"]] for x in matrix
            ]
            syntax_raw = decision(syntax_model, syntax_values)
            for row, value in zip(rows, syntax_raw, strict=True):
                row["syntax_order_raw"] = float(value)
                row["shortlist_score"] = float(
                    1 / (1 + math.exp(-max(-30.0, min(30.0, value))))
                )
                row["selection_score"] = row["shortlist_score"]
            research_used = True
        if ranker == "research" and rows:
            from .inclusion import InclusionModel

            learned = InclusionModel(self.config, self.lex, self.dialect)
            scores = learned.predict(rows, entry, matrix)
            preference = learned.predict_order(rows, entry, matrix)
            for row, value, rank_score in zip(rows, scores, preference, strict=True):
                row["learned_inclusion_score"] = float(value)
                row["shortlist_score"] = float(rank_score)
                row["selection_score"] = float(rank_score)
                row["learned_order_score"] = float(rank_score)
                row["sense_status"] = (
                    "learned_dictionary_conditioned_inclusion; expert token review pending"
                )
            inclusion_metadata = learned.metadata
            learned.close()
            research_used = True
        if ranker == "research":
            from .editorial import order

            rows = order(rows)
        else:
            rows.sort(key=lambda r: (-r["shortlist_score"], r["id"]))
        # Main inference ranks learned inclusion. Historical comparison methods
        # retain their separate quality floor.
        acceptance_floor = (
            0.0
            if ranker in {"research", "definition_syntax"}
            else max((r["acceptance_score"] for r in rows), default=0.0) * 0.50
        )
        quality_rows = [r for r in rows if r["acceptance_score"] >= acceptance_floor]
        excluded["below_single_sentence_quality_floor"] = len(rows) - len(quality_rows)
        pool = quality_rows[: max(100, k)]
        if pool and ranker != "research":
            floor = pool[0]["shortlist_score"] * 0.65
            pool = [r for r in pool if r["shortlist_score"] >= floor]
        selected = (
            pool[:k]
            if ranker in {"research", "definition_syntax"}
            else choose(pool, k, "greedy")
        )
        definition_review.sort(
            key=lambda r: (not bool(r["definition_relevance"]["crum"]), r["id"])
        )
        result = dict(
            schema_version=2,
            TLA=entry.tla,
            lemma=entry.lemma,
            definition=entry.gloss,
            dialect=self.dialect,
            candidate_count=len(candidates),
            eligible_count=len(rows),
            quality_eligible_count=len(quality_rows),
            diversity_pool_count=len(pool),
            excluded=dict(excluded),
            model_status=dict(
                acceptance=self.meta["acceptance_status"],
                crum_style=self.meta["style_status"],
            ),
            no_embedding_model_loaded=True,
            no_llm_called=True,
            shortlist=selected,
            ranker=ranker if research_used or ranker == "research" else "cheap",
            definition_admitted_count=len(rows) if ranker != "research" else None,
            model_scored_count=len(rows),
            source_diagnostic_pass_count=sum(
                r["definition_relevance"]["status"] == "pass" for r in rows
            ),
            source_diagnostic_pending_count=sum(
                r["definition_relevance"]["status"] != "pass" for r in rows
            ),
            definition_review_count=len(definition_review),
            definition_review=definition_review[:20],
            ordering_policy=(
                "learn dictionary-conditioned inclusion from TAB and Crum"
                if ranker == "research"
                else (
                    "definition gate then UD-only Crum order"
                    if ranker == "definition_syntax"
                    else "definition gate then baseline ordering"
                )
            ),
            definition_gate_applied=ranker != "research",
            inclusion_model=(
                inclusion_metadata if ranker == "research" and rows else None
            ),
            shortlist_metrics=metrics(selected, pool),
            warnings=[
                "The default model learns inclusion from TAB labels and Crum positives. Semantic values are model inputs; source-rule admission remains a separate baseline.",
                "Readability diagnostics and syntax scores have separate roles; their sigmoid values are uncalibrated.",
                "Local three-example selection is an experiment; LLM selection is a later step.",
            ],
        )
        result["model_status"]["provisional"] = self.meta.get("provisional", False)
        if self.meta.get("provisional"):
            result["warnings"].append(
                "This dialect has too little labeled evidence for reliable validation; model results are provisional."
            )
        same_category = [
            e
            for e in self.lex.values()
            if e.tla != entry.tla
            and e.lemma == entry.lemma
            and e.pos == entry.pos
            and e.gloss != entry.gloss
        ]
        result["sense_review_required"] = bool(same_category) and ranker == "research"
        result["definition_admission_policy"] = (
            "Diagnostic only for research; baseline admission: Manual evidence, Crum-listed construction or definition citation, grammatical definition, exclusive lexical match, or cached cosine threshold plus competing-sense margin; applied before quality and syntax"
        )
        if result["sense_review_required"]:
            result["warnings"].append(
                "TOKEN_SENSE_UNRESOLVED: exact TT matching does not distinguish these same-category meanings; returned candidates require sense review."
            )
        if ranker in {"research", "quality_first"} and not research_used and rows:
            result["warnings"].append(
                "Research ranker is unavailable for this dialect; the provisional two-task baseline was used."
            )
        if len(selected) < k:
            result["warnings"].append(
                "The shortlist contains the available candidates after the selected retrieval and data-cleaning requirements."
            )
        return result


def export(config, result, folder="shortlists"):
    import html

    if folder not in {"shortlists", "research_shortlists"}:
        raise ValueError("Unknown shortlist export folder")
    output = config.reports / folder / result["dialect"] / result["TLA"]
    candidates = []
    for i, r in enumerate(result["shortlist"], 1):
        candidates.append(
            dict(
                candidate_id=r["id"],
                rank=i,
                coptic=r["sent"],
                translation=r["translation"],
                citation=dict(
                    urn=r["urn"],
                    chapter=r["chapter"],
                    verse=r["verse"],
                    source=r["source"],
                ),
                dialect=r["dialect_id"],
                learned_inclusion_score=r.get("learned_inclusion_score"),
                quality_score=r["acceptance_score"],
                crum_style_percentile=r["crum_style_percentile"],
                information_score=r["information_score"],
                combined_score=r["shortlist_score"],
                syntax_signature=r["syntax"]["signature"],
                dependency_annotation=r["syntax"]["annotation_status"],
                sense_status=r["sense_status"],
            )
        )
    selections = {}
    for method in ["rank", "kmeans", "coverage"]:
        selected = choose(result["shortlist"], 3, method)
        selections[method] = dict(
            candidate_ids=[r["id"] for r in selected],
            metrics=metrics(selected, result["shortlist"]),
            experimental_local_selection=True,
        )
    packet = dict(
        schema_version=2,
        TLA=result["TLA"],
        lemma=result["lemma"],
        definition=result["definition"],
        dialect=result["dialect"],
        ranker=result.get("ranker", "research"),
        inclusion_model=result.get("inclusion_model"),
        definition_gate_applied=result.get("definition_gate_applied"),
        intended_next_step="Future LLM may choose three using only these candidates; not run in this phase",
        criteria=dict(
            single_sentence=[
                "correct target sense",
                "informative target context",
                "self-contained",
                "readable and typical",
            ],
            set_of_three=[
                "different target-centered syntactic uses",
                "avoid redundant contexts",
                "never mix dialects",
            ],
        ),
        candidates=candidates,
        local_selection_experiments=selections,
        model_status=result["model_status"],
        no_llm_called=True,
        no_embedding_model_loaded=True,
        warnings=result["warnings"],
    )
    write_json(output / "packet.json", packet)
    content = [
        f"<h1>{html.escape(result['TLA'])} · {html.escape(result['lemma'])} · {html.escape(result['dialect'])}</h1>",
        "<p>Ten-candidate local shortlist. No LLM was called. Three-example sets below are local experiments.</p>",
    ]
    for c in candidates:
        content.append(
            f"<article><h2>#{c['rank']} · quality {c['quality_score']:.3f} · style {c['crum_style_percentile']:.3f}</h2><p class=coptic>{html.escape(c['coptic'])}</p><p>{html.escape(c['translation'])}</p><p>{html.escape(', '.join(c['syntax_signature']))}</p><small>{html.escape(c['citation']['urn'])} {c['citation']['chapter']}:{c['citation']['verse']}</small></article>"
        )
    content.append(
        "<h2>Local selection experiments</h2><pre>"
        + html.escape(json.dumps(selections, ensure_ascii=False, indent=2))
        + "</pre>"
    )
    document = (
        '<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content="width=device-width, initial-scale=1"><title>Coptic shortlist</title><style>body{font:16px/1.65 system-ui;max-width:1100px;margin:35px auto;padding:0 20px;color:#183044;background:#f5f7f8}article{padding:24px;background:white;border:1px solid #d8e1e6;margin:20px 0;border-radius:8px}.coptic{font:24px/1.8 serif}small{overflow-wrap:anywhere}pre{white-space:pre-wrap}</style>'
        + "".join(content)
        + "</html>"
    )
    (output / "review.html").write_text(document, encoding="utf-8")
    return packet
