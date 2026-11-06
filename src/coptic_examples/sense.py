"""Match dictionary entries to TT tokens."""

from .text import lemma_key
from functools import lru_cache


@lru_cache(maxsize=1)
def _lexical_conditions():
    import json
    from .config import PROJECT

    path = PROJECT / "lexical_conditions.json"
    return json.loads(path.read_text()) if path.exists() else {}


def lexical_condition(tla):
    return _lexical_conditions().get(tla)


@lru_cache(maxsize=1)
def _registry():
    from .dialects import Dialects

    return Dialects()


# Dictionary labels and corpus native tags encode distinctions lost in coarse UPOS.
NATIVE = {
    "C615": {"PREP"},
    "C616": {"CCIRC"},
    "C617": {"CREL"},
    "C618": {"CFOC"},
    "C2786": {"PPOS"},
    "C2787": {"PPOS"},
    "C2788": {"COP"},
    "C2789": {"PTC"},
    "C773": {"PTC"},
    "C4008": {"PPERS", "PPERO", "PPERI"},
    "C614": {"NUM"},
}


def targets(candidate, entry):
    override = candidate.get("target_alignment", {}).get(entry.tla) or candidate.get(
        "crum_pos_alignment", {}
    ).get(entry.tla)
    if override:
        return [i for i, t in enumerate(candidate["tokens"]) if t["id"] in override]
    condition = lexical_condition(entry.tla)
    if condition:
        return [
            i
            for i, t in enumerate(candidate["tokens"])
            if t["lemma"] in condition["forms"]
            or lemma_key(t["form"]) in condition["forms"]
        ]
    registry = _registry()
    allcodes = {v for codes in (entry.form_dialects or {}).values() for v in codes}
    keys = {
        form
        for form in entry.forms
        if not ((entry.form_dialects or {}).get(form) or allcodes)
        or candidate.get("dialect_id")
        in registry.from_codes((entry.form_dialects or {}).get(form) or allcodes)
    }
    native = NATIVE.get(entry.tla)
    hits = []
    for form in keys:
        parts = form.split()
        if len(parts) < 2:
            continue
        for start in range(len(candidate["tokens"]) - len(parts) + 1):
            span = candidate["tokens"][start : start + len(parts)]
            if entry.pos and span[0]["pos"] != entry.pos:
                continue
            if all(
                part in {t["lemma"], lemma_key(t["form"])}
                for part, t in zip(parts, span)
            ):
                hits.extend(range(start, start + len(parts)))
    for i, t in enumerate(candidate["tokens"]):
        if t["lemma"] not in keys and lemma_key(t["form"]) not in keys:
            continue
        if native:
            if t["xpos"] not in native:
                continue
        elif entry.pos and t["pos"] != entry.pos:
            continue
        else:
            if not entry.pos:
                continue  # unresolved POS is not permission to match everything
        hits.append(i)
    return sorted(set(hits))


def align_candidate(candidate, entry, allow_pos_mismatch=False):
    import json
    from .config import PROJECT

    path = PROJECT / "token_alignments.json"
    overrides = json.loads(path.read_text()) if path.exists() else {}
    for record in overrides.get(entry.tla, []):
        if (
            candidate.get("source") != record["source"]
            or str(candidate.get("verse")) != record["verse"]
            or str(candidate.get("chapter")) != record["chapter"]
        ):
            continue
        pairs = [
            (i, t)
            for i, t in enumerate(candidate["tokens"])
            if t["id"] in record["token_ids"]
        ]
        if [t["id"] for i, t in pairs] != record["token_ids"] or [
            t["form"] for i, t in pairs
        ] != record["expected_forms"]:
            continue
        if [i for i, t in pairs] != list(range(pairs[0][0], pairs[-1][0] + 1)):
            continue
        candidate.setdefault("target_alignment", {})[entry.tla] = record["token_ids"]
        candidate["token_alignment_reference"] = record["reference"]
    indices = targets(candidate, entry)
    if (
        not indices
        and allow_pos_mismatch
        and entry.tla not in NATIVE
        and not lexical_condition(entry.tla)
    ):
        registry = _registry()
        keys = {
            f
            for f in entry.forms
            if not (entry.form_dialects or {}).get(f)
            or candidate.get("dialect_id")
            in registry.from_codes(entry.form_dialects[f])
        }
        indices = [i for i, t in enumerate(candidate["tokens"]) if t["lemma"] in keys]
        if not indices:
            indices = [
                i
                for i, t in enumerate(candidate["tokens"])
                if lemma_key(t["form"]) in keys and not t["lemma"]
            ]
        if indices:
            candidate.setdefault("target_alignment", {})[entry.tla] = [
                candidate["tokens"][i]["id"] for i in indices
            ]
    if indices:
        candidate["target_token_evidence"] = [candidate["tokens"][i] for i in indices]
        candidate["target_pos_mismatch"] = any(
            candidate["tokens"][i]["pos"] != entry.pos for i in indices
        )
    return indices


def native_candidates(config, corpus, entry, dialect, allow_pos_mismatch=False):
    import json, sqlite3
    from .config import database

    path = config.work / "cache/native_tokens.sqlite3"
    with corpus.connect() as source:
        stamp = source.execute(
            "SELECT value FROM metadata WHERE key='fingerprint'"
        ).fetchone()[0]
    with database(path) as db:
        db.execute("CREATE TABLE IF NOT EXISTS metadata(stamp TEXT)")
        row = db.execute("SELECT stamp FROM metadata").fetchone()
        if not row or row[0] != stamp:
            db.executescript(
                "DROP TABLE IF EXISTS token; CREATE TABLE token(sentence_id TEXT,form TEXT,lemma TEXT,dialect TEXT);"
            )
            for c in corpus.all():
                db.executemany(
                    "INSERT INTO token VALUES(?,?,?,?)",
                    [
                        (c["id"], lemma_key(t["form"]), t["lemma"], c["dialect_id"])
                        for t in c["tokens"]
                    ],
                )
            db.executescript(
                "CREATE INDEX native_form ON token(form,dialect); CREATE INDEX native_lemma ON token(lemma,dialect); DELETE FROM metadata;"
            )
            db.execute("INSERT INTO metadata VALUES(?)", (stamp,))
        condition = lexical_condition(entry.tla)
        keys = sorted(
            set(condition["forms"]) if condition else set(entry.forms) | {entry.lemma}
        )
        ids = set()
        for key in keys:
            key = key.split()[0] if " " in key else key
            ids.update(
                r[0]
                for r in db.execute(
                    "SELECT sentence_id FROM token WHERE form=? AND dialect=? UNION SELECT sentence_id FROM token WHERE lemma=? AND dialect=?",
                    (key, dialect, key, dialect),
                )
            )
    # Explicit, uniquely bound Crum citations may supply alignment when native POS disagrees.
    cited = set()
    if not condition:
        with database(config.work / "cache/crum_bible.sqlite3") as audit:
            cited = {
                r[0]
                for r in audit.execute(
                    "SELECT l.sentence_id FROM occurrence o JOIN link l ON o.id=l.occurrence_id WHERE l.dialect=? AND l.status='verified_target_form' AND json_array_length(o.payload,'$.tla_candidates')=1 AND json_extract(o.payload,'$.tla_candidates[0]')=?",
                    (dialect, entry.tla),
                )
            }
    ids.update(cited)
    candidates = []
    with corpus.connect() as db:
        for sid in sorted(ids):
            c = json.loads(
                db.execute(
                    "SELECT payload FROM sentence WHERE id=?", (sid,)
                ).fetchone()[0]
            )
            ii = align_candidate(c, entry, allow_pos_mismatch)
            if not ii and sid in cited:
                ii = [i for i, t in enumerate(c["tokens"]) if t["lemma"] in entry.forms]
                if not ii:
                    ii = [
                        i
                        for i, t in enumerate(c["tokens"])
                        if lemma_key(t["form"]) in entry.forms
                    ]
                if ii:
                    c["crum_pos_alignment"] = {
                        entry.tla: [c["tokens"][i]["id"] for i in ii]
                    }
            if ii:
                c["target_token_evidence"] = [c["tokens"][i] for i in ii]
                candidates.append(c)
                if condition:
                    c["lexical_condition"] = condition
    return candidates


def sense_text(candidate, entry):
    """Target-conditioned local contexts; English context is auxiliary evidence."""
    import re

    features = []
    for word in re.findall(r"[a-z]{2,}", candidate["translation"].lower()):
        features.append(entry.tla + "__en_" + word)
    for index in targets(candidate, entry):
        t = candidate["tokens"][index]
        features.extend(
            [entry.tla + "__xpos_" + t["xpos"], entry.tla + "__dep_" + t["dep"]]
        )
        for j in range(max(0, index - 3), min(len(candidate["tokens"]), index + 4)):
            if j != index:
                features.append(entry.tla + "__ctx_" + candidate["tokens"][j]["lemma"])
    return " ".join(features) or entry.tla + "__no_context"
