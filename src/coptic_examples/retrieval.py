"""All-corpus candidates + transparent sense contrast + learned example quality."""

import json
from dataclasses import asdict
import numpy as np
import lightgbm as lgb
from .annotations import forbidden_rules, forbidden, load_manual
from .corpus import Corpus, fingerprint
from .dataset import read_rows, text_key
from .embeddings import Embeddings
from .features import Features, quality
from .lexicon import load
from .text import (
    norm,
    lemma_key,
    pos,
    real_translation,
    sm_freq,
    rouge,
    editorial_heading,
)
from .dialects import Dialects


class AmbiguousLemma(ValueError):
    def __init__(self, query, entries):
        self.query = query
        self.entries = [asdict(e) for e in entries]
        super().__init__("Choose a dictionary entry for " + query)


class Search:
    def __init__(self, config):
        self.config = config
        self.corpus = Corpus(config)
        self.lex = load(config.lexicon)
        self.embeddings = None
        self.features = None
        self.annotations = load_manual(config.manual)
        self.rules = forbidden_rules(self.annotations)
        self.manual = {}
        for r in read_rows(config):
            if r["evidence"] == "manual":
                self.manual.setdefault(r["TLA"], []).append(r)

    def resolve(self, query, upos=None, definition=None):
        if query.upper() in self.lex:
            return self.lex[query.upper()]
        found = [
            e
            for e in self.lex.values()
            if lemma_key(query) in e.forms and (not upos or e.pos == pos(upos))
        ]
        if len(found) != 1:
            if found:
                raise AmbiguousLemma(query, found)
            raise ValueError("No dictionary entry found for " + query)
        return found[0]

    def search(
        self,
        query,
        top_n=10,
        ranker="research",
        semantic_limit=0,
        upos=None,
        definition=None,
        dialect="sahidic",
    ):
        if not 1 <= top_n <= 1000:
            raise ValueError("top_n must be in 1..1000")
        if semantic_limit < 0:
            raise ValueError("semantic_limit cannot be negative")
        dialect = Dialects().require(dialect)
        e = self.resolve(query, upos, definition)
        gloss = norm(definition) or e.gloss
        if ranker in {
            "cheap",
            "learned",
            "research",
            "unified",
            "quality_first",
            "definition_syntax",
        }:
            if definition:
                raise ValueError(
                    "Cheap models use the dictionary definition; custom definitions require explicit semantic review"
                )
            from .shortlist import Shortlist

            short = Shortlist(self.config, dialect).search(e.tla, top_n, ranker=ranker)
            rows = [
                dict(r, score=r["shortlist_score"], quality_score=r["acceptance_score"])
                for r in short["shortlist"]
            ]
            return dict(
                entry=asdict(e),
                dialect=dialect,
                definition=gloss,
                ranker=short["ranker"],
                requested_ranker=ranker,
                candidate_count=short["candidate_count"],
                semantic_scored_count=0,
                sense_eligible_count=short["eligible_count"],
                excluded=short["excluded"],
                results=rows,
                manual_examples=[
                    r
                    for r in self.manual.get(e.tla, [])
                    if r.get("dialect_id") == dialect
                ],
                warnings=short["warnings"],
                shortlist_evidence=short,
            )
        if self.embeddings is None:
            self.embeddings = Embeddings(self.config.vectors)
        self.features = Features(
            json.loads(
                (self.config.work / "models" / dialect / "frequencies.json").read_text()
            )
        )
        if not gloss:
            raise ValueError("An English definition is required")
        with self.corpus.connect() as db:
            candidates = [
                json.loads(x[0])
                for x in db.execute(
                    "SELECT s.payload FROM candidate c JOIN sentence s ON s.id=c.sentence_id WHERE c.tla=? AND c.dialect=? ORDER BY s.id",
                    (e.tla, dialect),
                )
            ]
        manual = [
            r for r in self.manual.get(e.tla, []) if r.get("dialect_id") == dialect
        ]
        total = len(candidates)
        manual_ids = {r["candidate"]["id"] for r in manual}
        eligible = []
        seen = set()
        counts = dict(
            no_translation=0,
            editorial_heading=0,
            forbidden=0,
            duplicates=0,
            sense_conflict=0,
        )
        for c in candidates:
            if forbidden(c, self.rules[e.tla], c["id"] in manual_ids):
                counts["forbidden"] += 1
                continue
            if not real_translation(c["translation"]):
                counts["no_translation"] += 1
                continue
            if editorial_heading(c["translation"]):
                counts["editorial_heading"] += 1
                continue
            key = (text_key(c), norm(c["translation"]))
            if key in seen:
                counts["duplicates"] += 1
                continue
            seen.add(key)
            eligible.append(c)
        if semantic_limit:
            eligible.sort(key=lambda c: (-rouge(gloss, c["translation"]), c["id"]))
            eligible = eligible[:semantic_limit]
        if not eligible:
            return dict(
                entry=asdict(e),
                dialect=dialect,
                candidate_count=total,
                semantic_scored_count=0,
                excluded=counts,
                results=[],
                manual_examples=manual,
                warnings=["No eligible translated examples in the requested dialect"],
            )
        q = self.embeddings.encode([gloss], "definition")[0]
        trans = [norm(c["translation"]) for c in eligible]
        v = self.embeddings.encode(trans)
        own = v @ q
        # Only this pair has been checked against expert anchors in this rebuild.
        other = {"C998": "C992", "C992": "C998"}.get(e.tla)
        competing = (
            v @ self.embeddings.encode([self.lex[other].gloss], "definition")[0]
            if other and not definition
            else None
        )
        rows = []
        matrix = []
        for i, c in enumerate(eligible):
            overlap = sm_freq(gloss, c["translation"])
            f = self.features.extract(c, e.lemma)
            f.update(sm_freq=overlap, sm_cosine=float(own[i]))
            row = dict(c)
            row.update(
                cosine=float(own[i]),
                sm_freq=overlap,
                rouge_l=rouge(gloss, c["translation"]),
                quality_score=quality(f),
                sense_status="not_contrast_validated",
            )
            if competing is not None:
                both = (
                    overlap > 0 and sm_freq(self.lex[other].gloss, c["translation"]) > 0
                )
                row.update(
                    competing_tla=other,
                    competing_cosine=float(competing[i]),
                    sense_margin=float(own[i] - competing[i]),
                )
                if both or own[i] <= competing[i]:
                    counts["sense_conflict"] += 1
                    continue
                row["sense_status"] = "automatic_contrast_support"
            row["linear_score"] = (
                0.55 * float(own[i])
                + 0.30 * row["rouge_l"]
                + 0.15 * row["quality_score"]
            )
            rows.append(row)
            matrix.append(f)
        if ranker == "linear":
            for r in rows:
                r["score"] = r["linear_score"]
        elif rows:
            raise ValueError("Unknown ranker")
        rows.sort(key=lambda r: (-r["score"], -r["cosine"], r["id"]))
        warnings = [
            "Sense decisions are automatic translation-based evidence, not token-level human annotation."
        ]
        if semantic_limit:
            warnings.append(
                f"Semantic scoring limited to {semantic_limit} lexical candidates; retrieval itself covered the whole corpus."
            )
        return dict(
            entry=asdict(e),
            dialect=dialect,
            definition=gloss,
            ranker=ranker,
            candidate_count=total,
            semantic_scored_count=len(eligible),
            sense_eligible_count=len(rows),
            excluded=counts,
            results=rows[:top_n],
            manual_examples=manual,
            warnings=warnings,
        )
