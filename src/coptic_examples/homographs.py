"""Exact spelling collisions with explicit dialect and lexical evidence."""

from collections import defaultdict
from .text import lemma_key, pos

DIALECTS = {"S": "sahidic", "Sa": "sahidic", "Sf": "sahidic", "B": "bohairic"}


def gloss(record):
    text = " | ".join(
        dict.fromkeys(
            g
            for s in record.get("senses", [])
            for g in s.get("translations", {}).get("en", [])
        )
    )
    if not text:
        grammar = {
            "Umstandsatzkonverter": "circumstantial converter",
            "Relativkonverter": "relative converter",
            "Fokalisierungskonverter": "focalising converter",
        }
        text = grammar.get(record.get("subc", ""), "")
    if "Titel" in str(record.get("subc", "")):
        text += " (title)"
    return text


def homograph_groups(records, canonical_only=True):
    groups = defaultdict(dict)
    for record in records:
        canonical = lemma_key(record.get("lemma", "")).casefold()
        forms = defaultdict(set)
        codes = defaultdict(set)
        for form in record.get("forms", []):
            key = lemma_key(form.get("orth", "")).casefold()
            if not key or (canonical_only and key != canonical):
                continue
            for code in str(form.get("dialect", "")).split("/"):
                if code in DIALECTS:
                    forms[key].add(DIALECTS[code])
                    codes[key].add(code)
        if canonical_only and canonical:
            forms.setdefault(canonical, set())
        for form, dialects in forms.items():
            groups[form][record["id"]] = dict(
                tla=record["id"],
                lemma=record.get("lemma", ""),
                pos=record.get("pos", ""),
                normalized_pos=pos(record.get("pos", "")),
                subcategory=record.get("subc", ""),
                gloss=gloss(record),
                dialects=sorted(dialects),
                dialect_codes=sorted(codes[form]),
            )
    result = []
    for form, entries in groups.items():
        values = sorted(entries.values(), key=lambda e: int(e["tla"][1:]))
        if len(values) < 2:
            continue
        shared = {
            d: [e["tla"] for e in values if d in e["dialects"]]
            for d in sorted(set(DIALECTS.values()))
        }
        shared = {d: ids for d, ids in shared.items() if len(ids) >= 2}
        pairs = [
            (a, b)
            for i, a in enumerate(values)
            for b in values[i + 1 :]
            if set(a["dialects"]) & set(b["dialects"])
        ]
        if not shared:
            category = "cross_dialect_or_unresolved"
        elif any(a["pos"] != b["pos"] for a, b in pairs):
            category = "grammatical_and_possible_semantic"
        elif any(a["gloss"].casefold() != b["gloss"].casefold() for a, b in pairs):
            category = "same_pos_different_gloss"
        else:
            category = "same_gloss_review_required"
        result.append(
            dict(
                word=form,
                category=category,
                shared_dialects=shared,
                senses=values,
                expert_verified=False,
                automatic_negative_labels=False,
            )
        )
    return sorted(result, key=lambda g: g["word"])
