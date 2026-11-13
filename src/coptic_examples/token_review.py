"""Apply local, context-specific corrections without changing corpus annotations."""

import hashlib
import json
from functools import lru_cache

from .config import PROJECT
from .text import norm


def text_digest(candidate):
    return hashlib.sha256(norm(candidate["sent"]).casefold().encode()).hexdigest()


@lru_cache(maxsize=1)
def records():
    path = PROJECT / "token_sense_reviews.json"
    grouped = {}
    if path.exists():
        for row in json.loads(path.read_text()):
            key = (row["dialect"], row["text_sha256"])
            grouped.setdefault(key, []).append(row)
    return grouped


def decisions(candidate, tla):
    """Return decisions only when text, token position and form still agree."""
    if not candidate.get("sent") or not candidate.get("dialect_id"):
        return {}
    rows = records().get((candidate.get("dialect_id"), text_digest(candidate)), [])
    out = {}
    for row in rows:
        if tla not in row["scope"]:
            continue
        i = row["token_index"]
        if i >= len(candidate["tokens"]):
            continue
        token = candidate["tokens"][i]
        if token["form"] != row["expected_form"]:
            continue
        if token["lemma"] != row["expected_lemma"]:
            continue
        out[i] = tla in row["accepted"]
    return out


def apply(candidate, entry, indices):
    reviewed = decisions(candidate, entry.tla)
    return sorted(
        {i for i in indices if reviewed.get(i, True)}
        | {i for i, accepted in reviewed.items() if accepted}
    )
