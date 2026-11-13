"""Whole-form Sahidic spelling compatibility from a traceable normalization table."""

from collections import defaultdict
import hashlib
from pathlib import Path
from .text import lemma_key

VOWELS = set("ⲁⲉⲏⲓⲟⲱⲩ")
SACRED_ABBREVIATIONS = {"ⲓⲥ": "ⲓⲏⲥⲟⲩⲥ", "ⲭⲥ": "ⲭⲣⲓⲥⲧⲟⲥ"}


def compatible_spelling(original, normalized):
    if original == normalized:
        return True
    if SACRED_ABBREVIATIONS.get(original) == normalized:
        return True
    # Consonant substitutions and longer rewrites require separate witnesses.
    consonants = lambda value: "".join(c for c in value if c not in VOWELS)
    if min(len(original), len(normalized)) < 3 or consonants(original) != consonants(
        normalized
    ):
        return False
    distance = list(range(len(normalized) + 1))
    for i, a in enumerate(original, 1):
        following = [i]
        for j, b in enumerate(normalized, 1):
            following.append(
                min(following[-1] + 1, distance[j] + 1, distance[j - 1] + (a != b))
            )
        distance = following
    return distance[-1] <= 2


def spelling_key(text):
    return lemma_key(text.casefold()).replace(" ", "")


class SpellingCompatibility:
    def __init__(self, table):
        table = Path(table)
        self.sha256 = hashlib.sha256(table.read_bytes()).hexdigest()
        alternatives = defaultdict(set)
        for line in table.read_text().splitlines():
            fields = line.split("\t")
            if len(fields) < 2 or any(c in fields[0] + fields[1] for c in "_| "):
                continue
            original, normalized = map(spelling_key, fields[:2])
            if original and normalized:
                alternatives[original].add(normalized)
        self.mapping = {
            original: next(iter(values))
            for original, values in alternatives.items()
            if len(values) == 1
            and len(original) >= 2
            and compatible_spelling(original, next(iter(values)))
        }
        self.mapping["ⲉⲣ"] = "ⲉⲣ"
        self.ambiguous = sum(len(v) > 1 for v in alternatives.values())

    def key(self, text):
        value = spelling_key(text)
        return self.mapping.get(value, value)
