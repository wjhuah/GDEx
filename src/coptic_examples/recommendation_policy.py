"""Recommendation length and reviewed dialect restrictions."""

import json
import time
import unicodedata
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from .text import COPTIC

MAX_LETTERS = 30
REMOTE = "https://raw.githubusercontent.com/wjhuah/GDEx/main/dialect_bias.md"
_checked = {}


def coptic_letters(text):
    return sum(unicodedata.category(c).startswith("L") for c in COPTIC.findall(text))


def parse_bias(text):
    rows = [
        r.strip().strip("|").split("|")
        for r in text.splitlines()
        if r.strip().startswith("|")
    ]
    if not rows:
        raise ValueError("Missing dialect table")
    header = [x.strip() for x in rows[0]]
    if header[:2] != ["TLA ID", "Lemma"]:
        raise ValueError("Expected TLA ID and Lemma")
    out = {}
    for row in rows[2:]:
        row = [x.strip() for x in row]
        if (
            len(row) != len(header)
            or row[0] in out
            or not row[0].startswith("C")
            or not row[0][1:].isdigit()
        ):
            raise ValueError("Invalid dialect row")
        flags = {}
        for name, value in zip(header[2:], row[2:]):
            if not name.endswith(" rare") or value.upper() not in {"T", "F"}:
                raise ValueError("Use dialect rare columns and T/F values")
            flags[name[:-5].casefold()] = value.upper() == "T"
        out[row[0]] = flags
    return out


def dialect_bias(config):
    local = config.work / "dialect_bias.md"
    baseline = parse_bias(local.read_text()) if local.exists() else {}
    cache = config.work / "cache/dialect_bias_remote.json"
    key = str(config.work)
    if time.monotonic() - _checked.get(key, -1000) > 60:
        _checked[key] = time.monotonic()
        try:
            with urlopen(
                Request(REMOTE, headers={"User-Agent": "GDEx"}), timeout=2
            ) as response:
                remote = parse_bias(response.read(100000).decode("utf-8"))
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".tmp")
            tmp.write_text(json.dumps(remote))
            tmp.replace(cache)
        except HTTPError as exc:
            exc.close()
        except (OSError, ValueError, UnicodeError):
            pass
    if cache.exists():
        try:
            baseline.update(json.loads(cache.read_text()))
        except (ValueError, OSError):
            pass
    return baseline


def exclusion_reason(config, tla, dialect, candidate, rules=None):
    if (
        (dialect_bias(config) if rules is None else rules)
        .get(tla, {})
        .get(dialect.casefold(), False)
    ):
        return "reviewed_dialect_restriction"
    if coptic_letters(candidate["sent"]) > MAX_LETTERS:
        return "over_30_coptic_letters"
    return None
