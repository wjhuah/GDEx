"""Evidence-based checks for damaged original spans paired with long translations."""

import re
import json
from functools import lru_cache
from .config import PROJECT


@lru_cache(maxsize=1)
def source_issues():
    path = PROJECT / "source_integrity.json"
    return json.loads(path.read_text()) if path.exists() else {}


def alignment_issue(candidate):
    recorded = source_issues().get(candidate.get("source"))
    if recorded and candidate.get("supplemental_source", {}).get("sha256") == recorded["sha256"]:
        return recorded["issue"]
    damaged = candidate.get("has_lacuna", False) or any(
        "[" in g or "]" in g or "…" in g for g in candidate.get("original_groups", [])
    )
    count = len(candidate.get("tokens", []))
    english = len(re.findall(r"[A-Za-z]+", candidate.get("translation", "")))
    if damaged and count <= 4 and english >= 10 and english > 4 * count:
        return "lacuna_fragment_with_full_translation"
    return None
