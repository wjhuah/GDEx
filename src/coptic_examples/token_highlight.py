"""Align the complete TT token stream to bound-group display text."""

import html
from .text import norm


def spans(text, tokens):
    positions = [i for i, c in enumerate(text) if not c.isspace()]
    compact = "".join(text[i] for i in positions)
    forms = ["".join(norm(t["form"]).split()) for t in tokens]
    if "".join(forms) != compact:
        return None
    result = []
    offset = 0
    for t, form in zip(tokens, forms):
        if form:
            result.append(
                (t["id"], positions[offset], positions[offset + len(form) - 1] + 1)
            )
        offset += len(form)
    return result


def render(text, tokens, target_ids):
    aligned = spans(text, tokens)
    ids = set(target_ids)
    if aligned is None:
        return (
            html.escape(text)
            + "<small> · TT alignment review: "
            + html.escape(
                " | ".join(t["id"] + ": " + t["form"] for t in tokens if t["id"] in ids)
            )
            + "</small>"
        )
    output = []
    cursor = 0
    for tid, start, end in aligned:
        if tid not in ids:
            continue
        output.append(html.escape(text[cursor:start]))
        output.append(
            '<mark class="target-token" data-token-id="'
            + html.escape(tid)
            + '" title="TT '
            + html.escape(tid)
            + '">'
            + html.escape(text[start:end])
            + "</mark>"
        )
        cursor = end
    output.append(html.escape(text[cursor:]))
    return "".join(output)
