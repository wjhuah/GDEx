"""Local printed scope of rarity qualifiers, without inheriting a dialect ban."""

import re


def rarity_scope(before, after):
    direct = re.search(
        r"(?P<form>⟨[^⟩]+⟩)?\s*(?P<codes>(?:⟪[^⟫]+⟫\s*)+)\((?:very\s+|extremely\s+|both\s+)?$",
        before,
        re.I,
    )
    if direct:
        return dict(
            scope="方言词形",
            dialect_codes=re.findall(r"⟪([^⟫]+)⟫", direct["codes"]),
            form=direct["form"] or "",
            evidence=direct[0],
            rule="dialect_before_parenthesis",
        )
    following = re.match(r"\s*\)\s*((?:⟪[^⟫]+⟫\s*)+)", after)
    form = re.search(r"⟨([^⟩]+)⟩\s*\((?:very\s+)?$", before)
    if form:
        return dict(
            scope="方言词形" if following else "词形",
            dialect_codes=re.findall(r"⟪([^⟫]+)⟫", following[1]) if following else [],
            form=form[1],
            evidence=form[0] + ("…" + following[0] if following else ""),
            rule="form_before_parenthesis",
        )
    tags = re.search(r"⟦([^⟧]+)⟧\{annotation:[^}]*\}\s*\((?:very\s+)?$", before)
    if tags:
        scope = {
            "tr": "及物用法",
            "intr": "不及物用法",
            "sg": "数的用法",
            "pl": "数的用法",
            "qual": "状态形式用法",
        }.get(tags[1].lower(), "语法用法")
        # Singular used as plural has its own scope rather than a word-wide ban.
        if re.search(r"⟦sg⟧.*as.*⟦pl⟧", before):
            scope = "单数形式用作复数"
        return dict(
            scope=scope,
            dialect_codes=[],
            form="",
            evidence=tags[0],
            rule="grammatical_label",
        )
    if re.search(r"⟨gloss:[^⟩]+⟩\s*\($", before) or re.match(r"\s*⟨gloss:", after):
        return dict(
            scope="特定义项",
            dialect_codes=[],
            form="",
            evidence=before[-100:] + "…" + after[:100],
            rule="gloss",
        )
    if re.match(r"\s*⟦(?:sg|pl)⟧", after):
        return dict(
            scope="数的用法",
            dialect_codes=[],
            form="",
            evidence=after[:120],
            rule="following_number",
        )
    if re.match(
        r"\s*(?:without|with|before|after|followed|following|but)\b", after, re.I
    ) or re.search(
        r"\b(?:with|without|following|followed|constructed)\b", before[-160:], re.I
    ):
        return dict(
            scope="局部构式或搭配",
            dialect_codes=[],
            form="",
            evidence=before[-160:] + "…" + after[:160],
            rule="construction",
        )
    if re.match(r"\s*⟦c⟧", after) or re.match(r"\s*⟨[^⟩]+⟩", after):
        return dict(
            scope="后接构式或变化词形",
            dialect_codes=[],
            form="",
            evidence=after[:160],
            rule="following_form",
        )
    return dict(
        scope="待核定",
        dialect_codes=[],
        form="",
        evidence=before[-160:] + "…" + after[:160],
        rule="unresolved",
    )
