"""Dictionary-conditioned inclusion prediction; source evidence supplies training labels."""

import math, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .ablation import CachedVectors
from .features import Features
from .shortlist import numbers, COLUMNS
from .text import sm_freq, rouge
from .relevance import bleu2

SEMANTIC = [
    "sense_overlap",
    "sense_cosine",
    "sense_cosine_missing",
    "sense_rouge",
    "sense_bleu2",
    "rival_overlap_max",
    "sense_overlap_margin",
    "rival_cosine_max",
    "sense_cosine_margin",
    "rival_count",
    "bound_personal_pronoun_ratio",
    "independent_personal_pronoun_ratio",
]
ALIGNMENT = ["target_pos_match_ratio", "target_pos_conflict"]
from .typicality import COLUMNS as TYPICALITY

COLUMNS_INCLUSION = COLUMNS + SEMANTIC + ALIGNMENT + TYPICALITY


class EntryFeatures:
    def __init__(self, config, lex, dialect):
        self.vectors = CachedVectors(config)
        self.lex = lex
        self.dialect = dialect
        self.extractor = Features(
            json.loads(
                (config.work / "models" / dialect / "frequencies.json").read_text()
            )
        )
        self.byform = defaultdict(set)
        for e in lex.values():
            for form in e.forms:
                self.byform[form].add(e.tla)
        from .typicality import Typicality

        self.typicality = Typicality(config, dialect)
        self.memo = {}

    def rivals(self, entry):
        ids = set().union(*(self.byform[f] for f in entry.forms))
        return [
            self.lex[t]
            for t in sorted(ids)
            if t != entry.tla and self.lex[t].gloss != entry.gloss
        ]

    def vector(self, c, e, base=None):
        key = (c["id"], e.tla)
        if key in self.memo:
            return self.memo[key]
        base = base if base is not None else numbers(c, e, self.extractor)[0]
        rivals = self.rivals(e)
        overlap = sm_freq(e.gloss, c["translation"])
        co = self.vectors.cosine(e.gloss, c["translation"])
        ro = max((sm_freq(v.gloss, c["translation"]) for v in rivals), default=0.0)
        cs = [self.vectors.cosine(v.gloss, c["translation"]) for v in rivals]
        available = [v for v in cs if v is not None]
        rc = max(available) if available else None
        # Missing rival vectors preserve a missing margin rather than inventing a semantic difference.
        margin = (
            co - rc
            if co is not None and rc is not None and len(available) == len(cs)
            else (co if co is not None and not rivals else math.nan)
        )
        n = max(1, len(c["tokens"]))
        bound = (
            sum(
                t["xpos"] in {"PPERS", "PPERO"}
                or t["xpos"].endswith(("_PPERS", "_PPERO"))
                for t in c["tokens"]
            )
            / n
        )
        free = sum(t["xpos"] == "PPERI" for t in c["tokens"]) / n
        x = list(base) + [
            overlap,
            co if co is not None else math.nan,
            float(co is None),
            rouge(e.gloss, c["translation"]),
            bleu2(e.gloss, c["translation"]),
            ro,
            overlap - ro,
            rc if rc is not None else math.nan,
            margin,
            len(rivals),
            bound,
            free,
        ]
        from .sense import targets

        indices = targets(c, e) if all("lemma" in t for t in c["tokens"]) else []
        match = (
            sum(c["tokens"][i]["pos"] == getattr(e, "pos", "") for i in indices)
            / len(indices)
            if indices
            else 0.0
        )
        x.extend([match, float(bool(indices) and match < 1)])
        typicality = getattr(self, "typicality", None)
        x.extend(typicality.vector(c, e) if typicality else [0.0] * len(TYPICALITY))
        self.memo[key] = x
        return x

    def close(self):
        self.vectors.connection.close()


class InclusionModel:
    def __init__(self, config, lex, dialect):
        import lightgbm as lgb

        self.features = EntryFeatures(config, lex, dialect)
        folder = config.work / "models" / dialect
        self.folder = folder
        self.metadata = json.loads((folder / "inclusion_learned.json").read_text())
        import hashlib

        if (
            self.metadata["manual_sha256"]
            != hashlib.sha256(config.manual.read_bytes()).hexdigest()
        ):
            raise RuntimeError("Rebuild inclusion model after annotation changes")
        if self.metadata["columns"] != COLUMNS_INCLUSION:
            raise RuntimeError("Rebuild inclusion model after feature changes")
        self.model = lgb.Booster(model_file=str(folder / self.metadata["file"]))

    def predict(self, rows, entry, bases):
        matrix = [
            self.features.vector(c, entry, b) for c, b in zip(rows, bases, strict=True)
        ]
        return self.model.predict(np.asarray(matrix), num_threads=4)

    def predict_order(self, rows, entry, bases):
        import lightgbm as lgb

        path = self.metadata.get("order_file")
        if not path:
            return self.predict(rows, entry, bases)
        model = lgb.Booster(model_file=str(self.folder / path))
        values = np.asarray(
            [
                self.features.vector(c, entry, b)
                for c, b in zip(rows, bases, strict=True)
            ]
        )
        raw = model.predict(values, num_threads=4)
        alpha = self.metadata.get("ranking_mix", 1.0)
        return (1 - alpha) * self.predict(rows, entry, bases) + alpha / (
            1 + np.exp(-np.clip(raw, -30, 30))
        )

    def close(self):
        self.features.close()
