"""Recover citation bindings from explicit article headings and native token spans."""

import re
from .crum_audit import heading_forms
from .text import lemma_key


def expanded_heading_forms(paragraph, root):
    forms = heading_forms(paragraph, root)
    # Bare abbreviated prefixes can also be independent dictionary words.
    # Prefer the complete derived form when its full spelling is reconstructed.
    header = paragraph.split("⟨gloss:", 1)[0].split("{bible:", 1)[0]
    short = {lemma_key(x) for x in re.findall(r"⟨([^⟩]+)⟩", header)}
    if root in short:
        return list(dict.fromkeys(f for f in forms if f in short))
    expanded = [f for f in forms if f not in short]
    return expanded or forms


def bind_forms(forms, page, lexicon, byform, pages):
    candidates = {
        t for f in forms for t in byform.get(f, []) if page in pages.get(t, set())
    }
    return sorted(candidates)


def token_spans(candidate, entry, printed_variants=(), normalizer=None):
    """Allow different TT splits only when their complete forms concatenate exactly."""
    key = (
        normalizer
        if normalizer is not None
        else lambda value: lemma_key(value).replace(" ", "")
    )
    keys = {key(f) for f in (*entry.forms, *printed_variants)}
    found = []
    tokens = candidate["tokens"]
    for token in tokens:
        if key(token.get("lemma", "")) in keys:
            found.append([token["id"]])
    for start in range(len(tokens)):
        spelling = ""
        for end in range(start, min(start + 4, len(tokens))):
            part = lemma_key(tokens[end]["form"]).replace(" ", "")
            if not part:
                break
            spelling += part
            if key(spelling) in keys:
                found.append([t["id"] for t in tokens[start : end + 1]])
    return [list(span) for span in dict.fromkeys(tuple(span) for span in found)]


def has_lexical_heading(paragraph):
    from .crum_audit import LEAD

    lead = LEAD.match(paragraph)
    return bool(lead and lemma_key(lead[1]))


def has_dialect_alternative(paragraph, raw, dialect_codes):
    """Identify explicitly bracketed dialect readings beside this occurrence."""
    following = paragraph.split(raw, 1)[-1]
    following = re.split(r"⟦;⟧|⟦[^⟧]*⟧\{(?:bible|reference):", following, maxsplit=1)[0]
    for bracket in re.findall(r"\(([^)]*)\)", following):
        for code in dialect_codes:
            parts = bracket.split("⟪" + code + "⟫", 1)
            if len(parts) == 2 and (
                lemma_key(parts[1]) or "annotation: ditto" in parts[1]
            ):
                return True
    return False


def explicit_dialect_forms(paragraph, raw, dialect_codes, root):
    """Read the quoted replacement spelling for a dialect at this citation."""
    following = paragraph.split(raw, 1)[-1]
    following = re.split(r"⟦;⟧|⟦[^⟧]*⟧\{(?:bible|reference):", following, maxsplit=1)[0]
    forms = []
    for bracket in re.findall(r"\(([^)]*)\)", following):
        for code in dialect_codes:
            if "⟪" + code + "⟫" not in bracket:
                continue
            local = bracket.split("⟪" + code + "⟫", 1)[1].split("⟪", 1)[0]
            quotes = re.findall(r"⟨([^⟩]+)⟩", local)
            for quote in quotes:
                if lemma_key(quote):
                    forms.extend(expanded_heading_forms("¶ ⟨" + quote + "⟩", root))
    return list(dict.fromkeys(forms))


def add_cdo_only_entries(lexicon, records):
    """Include online-only entries in a recovery lexicon without changing deployed inputs."""
    from .lexicon import Entry, apply_snapshot
    from .text import pos

    added = []
    for record in records:
        tla = record["id"]
        if tla in lexicon:
            continue
        raw_lemma = record.get("lemma", "")
        form = lemma_key(raw_lemma)
        if not form:
            continue
        lexicon[tla] = Entry(
            tla=tla,
            lemma=form,
            pos=pos(record.get("pos", "")),
            gloss="",
            forms=(form,),
            form_dialects={},
            gloss_source="https://coptic-dictionary.org/entry/" + tla,
        )
        added.append(tla)
    apply_snapshot(lexicon, [r for r in records if r["id"] in set(added)])
    return added


def contextual_dialect_reading(paragraph, raw, dialect_codes, root):
    """Resolve a bracketed spelling or ditto from earlier in the same paragraph."""
    tokens = re.finditer(r"⟦[^⟧]*⟧(?:\{[^}]*\})?|⟪[^⟫]+⟫|⟨[^⟩]+⟩|[()]", paragraph)
    remembered = {}
    depth = 0
    active_code = None
    target = False
    readings = []
    evidence = []
    for match in tokens:
        value = match[0]
        if value.startswith("⟦") and ("{bible:" in value or "{reference:" in value):
            if target:
                break
            target = value == raw
            active_code = None
            continue
        if value == "(":
            depth += 1
        elif value == ")":
            depth = max(0, depth - 1)
            active_code = None
        elif value.startswith("⟪"):
            active_code = value[1:-1] if depth else None
        elif active_code and value.startswith("⟨"):
            spelling = value[1:-1]
            if lemma_key(spelling):
                remembered[active_code] = (spelling, match.start())
                if target and active_code in dialect_codes:
                    readings.extend(
                        expanded_heading_forms("¶ ⟨" + spelling + "⟩", root)
                    )
                    evidence.append(
                        dict(kind="explicit", spelling=spelling, offset=match.start())
                    )
        elif active_code and "annotation: ditto" in value:
            if active_code in remembered:
                spelling, offset = remembered[active_code]
                if target and active_code in dialect_codes:
                    readings.extend(
                        expanded_heading_forms("¶ ⟨" + spelling + "⟩", root)
                    )
                    evidence.append(
                        dict(kind="ditto", spelling=spelling, antecedent_offset=offset)
                    )
        elif active_code and (
            "annotation: omits" in value or "annotation: different reading" in value
        ):
            remembered.pop(active_code, None)
    return list(dict.fromkeys(readings)), evidence


def explicit_bible_location(work, chapter, verse):
    """Use a fully stated verse label when the chapter metadata is a default."""
    from .text import location_key

    book = re.search(r":(?:ot|nt)\.([^.]+)", work)
    if not book:
        return None
    full = re.fullmatch(r".+?\s+(\d+):(\d+[a-z]?)", str(verse))
    if full:
        return book[1].lower(), location_key(full[1]), location_key(full[2])
    if str(chapter).isdigit() and re.fullmatch(r"\d+[a-z]?", str(verse)):
        return book[1].lower(), location_key(chapter), location_key(verse)
    return None


def has_government_alternative(paragraph, raw, dialect_codes):
    """Distinguish a short governed-preposition change from replacing the headword."""
    prefix = paragraph.split("{bible:", 1)[0]
    if has_lexical_heading(paragraph) or "annotation: constructed with" not in prefix:
        return False
    following = paragraph.split(raw, 1)[-1]
    following = re.split(r"⟦;⟧|⟦[^⟧]*⟧\{(?:bible|reference):", following, maxsplit=1)[0]
    readings = []
    for bracket in re.findall(r"\(([^)]*)\)", following):
        for code in dialect_codes:
            if "⟪" + code + "⟫" in bracket:
                local = bracket.split("⟪" + code + "⟫", 1)[1].split("⟪", 1)[0]
                readings.extend(re.findall(r"⟨([^⟩]+)⟩", local))
    return bool(readings) and all(
        q.strip().endswith("-") and 0 < len(lemma_key(q)) <= 2 for q in readings
    )
