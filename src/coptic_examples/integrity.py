"""Evidence-based checks for damaged original spans paired with long translations."""

import re


def alignment_issue(candidate):
    damaged = candidate.get("has_lacuna", False) or any(
        "[" in g or "]" in g or "…" in g for g in candidate.get("original_groups", [])
    )
    count = len(candidate.get("tokens", []))
    english = len(re.findall(r"[A-Za-z]+", candidate.get("translation", "")))
    if damaged and count <= 4 and english >= 10 and english > 4 * count:
        return "lacuna_fragment_with_full_translation"
    return None
