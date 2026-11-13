"""Publish an auditable homograph candidate list from the pinned online index."""

import sys, json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config
from coptic_examples.homographs import homograph_groups

cfg = Config()
records = json.loads((cfg.work / "cache/cdo_online/index.json").read_text())
root = cfg.reports / "本地复盘/数据规模与同形词"
root.mkdir(parents=True, exist_ok=True)
canonical = homograph_groups(records, True)
forms = homograph_groups(records, False)
focus = ["ⲉⲓⲱⲧ", "ϩⲟⲟⲩ", "ⲧⲱⲙⲧ", "ⲉϩⲉ", "ⲡⲉ", "ⲉ"]
canonical.sort(
    key=lambda g: (
        focus.index(g["word"]) if g["word"] in focus else len(focus),
        g["word"],
    )
)
summary = dict(
    dictionary_entries=len(records),
    canonical_families=len(canonical),
    canonical_entries=len({e["tla"] for g in canonical for e in g["senses"]}),
    canonical_within_dialect=sum(bool(g["shared_dialects"]) for g in canonical),
    canonical_categories=dict(Counter(g["category"] for g in canonical)),
    all_form_collision_groups=len(forms),
    all_form_within_dialect=sum(bool(g["shared_dialects"]) for g in forms),
    all_form_entries=len({e["tla"] for g in forms for e in g["senses"]}),
)
(root / "规模.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
)
(root / "同词头候选.json").write_text(
    json.dumps(canonical, ensure_ascii=False, indent=2) + "\n"
)
(root / "全部词形冲突.json").write_text(
    json.dumps(forms, ensure_ascii=False, indent=2) + "\n"
)


def escape(s):
    return s.replace("|", " / ").replace("\n", " ")


def sense(e):
    return escape(
        f"{e['tla']} · {e['pos']} · {','.join(e['dialect_codes']) or 'dialect unassigned'} · {e['gloss'] or '[English gloss unavailable]'}"
    )


lines = [
    "# Homograph vocabulary",
    "",
    f"Source: pinned CDO index, {len(records):,} entries. Exact complete spellings after case and editorial-mark normalization; spaces in multiword expressions are retained.",
    "",
    f"{len(canonical):,} canonical spelling families; {summary['canonical_within_dialect']:,} have at least two entries with explicitly shared Sahidic or Bohairic forms. All-form collisions are listed separately in the local report.",
    "",
    "This is a review candidate inventory. Distinct TLA IDs can reflect parts of speech, lexical senses, gender or other dictionary divisions. Shared spelling alone supplies no negative labels. Cross-dialect-only groups remain separate.",
    "",
    "| Word | Sense 1 | Sense 2 | Sense 3+ |",
    "| --- | --- | --- | --- |",
]
for g in canonical:
    if not g["shared_dialects"]:
        continue
    senses = g["senses"]
    other = "<br>".join(sense(e) for e in senses[2:])
    shared = ", ".join(g["shared_dialects"]) or "cross-dialect / review"
    lines.append(
        "| "
        + " | ".join(
            [g["word"] + "<br>" + shared, sense(senses[0]), sense(senses[1]), other]
        )
        + " |"
    )
lines += [
    "",
    "## Cross-dialect or unresolved appendix",
    "",
    "| Word | Sense 1 | Sense 2 | Sense 3+ |",
    "| --- | --- | --- | --- |",
]
for g in canonical:
    if g["shared_dialects"]:
        continue
    ss = g["senses"]
    lines.append(
        "| "
        + " | ".join(
            [
                g["word"],
                sense(ss[0]),
                sense(ss[1]),
                "<br>".join(sense(e) for e in ss[2:]),
            ]
        )
        + " |"
    )
(cfg.work / "homograph_vocabulary.md").write_text("\n".join(lines) + "\n")
form_lines = [
    "# Shared-form vocabulary",
    "",
    "Complete form collisions, including inflected and bound forms. Same-dialect groups are listed first; cross-dialect or unassigned groups form an appendix. These records supply review candidates, without automatic negative labels.",
    "",
    "| Word | Sense 1 | Sense 2 | Sense 3+ |",
    "| --- | --- | --- | --- |",
]
for group in sorted(forms, key=lambda g: (not bool(g["shared_dialects"]), g["word"])):
    ss = group["senses"]
    shared = ", ".join(group["shared_dialects"]) or "cross-dialect / review"
    form_lines.append(
        "| "
        + " | ".join(
            [
                group["word"] + "<br>" + shared,
                sense(ss[0]),
                sense(ss[1]),
                "<br>".join(sense(e) for e in ss[2:]),
            ]
        )
        + " |"
    )
(cfg.work / "homograph_forms.md").write_text("\n".join(form_lines) + "\n")
# Keep the hand-maintained regression vocabulary and source cases as the focused test list.
assert all(any(g["word"] == f for g in canonical) for f in focus)
print(json.dumps(summary, ensure_ascii=False, indent=2))
