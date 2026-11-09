#!/usr/bin/env python3
"""Describe aligned Crum spans, including within-entry standardized lengths."""

import sys, json, csv
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config, write_json
from coptic_examples.recommendation_policy import coptic_letters

cfg = Config()
folder = cfg.reports / "crum_lengths"
folder.mkdir(parents=True, exist_ok=True)
positions = json.loads((cfg.work / "cache/crum_source_positions.json").read_text())
groups = defaultdict(list)
rows = []
for line in (cfg.derived / "learning_examples.jsonl").open():
    r = json.loads(line)
    if r["y"] <= 0 or "crum" not in r["sources"]:
        continue
    c = r["candidate"]
    pp = [
        positions[x["occurrence_id"]]["position"]
        for x in r.get("crum_refs", [])
        if x.get("occurrence_id") in positions
    ]
    item = dict(
        dialect=r["dialect"],
        tla=r["tla"],
        sentence_id=c["id"],
        letters=coptic_letters(c["sent"]),
        tt_tokens=len(c["tokens"]),
        bound_groups=len(c["sent"].split()),
        first_citation_position=min(pp) if pp else None,
    )
    rows.append(item)
    groups[(r["dialect"], r["tla"])].append(item)
entries = []
for (d, t), rr in sorted(groups.items()):
    values = np.array([r["letters"] for r in rr])
    mean = float(values.mean())
    sd = float(values.std())
    for r in rr:
        r["within_entry_z"] = (r["letters"] - mean) / sd if sd else 0.0
    ordered = [r for r in rr if r["first_citation_position"] is not None]
    rho = None
    if (
        len(ordered) >= 3
        and len({r["letters"] for r in ordered}) > 1
        and len({r["first_citation_position"] for r in ordered}) > 1
    ):
        rho = float(
            spearmanr(
                [r["first_citation_position"] for r in ordered],
                [r["letters"] for r in ordered],
            ).statistic
        )
    entries.append(
        dict(
            dialect=d,
            tla=t,
            count=len(rr),
            mean=mean,
            std=sd,
            min=int(values.min()),
            median=float(np.median(values)),
            p90=float(np.quantile(values, 0.9)),
            max=int(values.max()),
            over_30=int(sum(values > 30)),
            order_length_spearman=rho,
        )
    )
summary = {}
for d in sorted({r["dialect"] for r in rows}):
    rr = [r for r in rows if r["dialect"] == d]
    a = np.array([r["letters"] for r in rr])
    rho = [
        e["order_length_spearman"]
        for e in entries
        if e["dialect"] == d and e["order_length_spearman"] is not None
    ]
    summary[d] = dict(
        aligned_entry_sentence_pairs=len(rr),
        entries=sum(e["dialect"] == d for e in entries),
        mean_letters=float(a.mean()),
        median_letters=float(np.median(a)),
        p90_letters=float(np.quantile(a, 0.9)),
        over_30=int(sum(a > 30)),
        at_most_30=int(sum(a <= 30)),
        entries_with_order_correlation=len(rho),
        median_within_entry_order_length_rho=float(np.median(rho)) if rho else None,
    )
write_json(
    folder / "summary.json",
    dict(
        raw_bible_markers=sum(
            1 for _ in (cfg.derived / "crum_bible_occurrences.jsonl").open()
        ),
        aligned=summary,
    ),
)
for name, values in [("entries", entries), ("examples", rows)]:
    with (folder / (name + ".csv")).open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
lines = [
    "# Crum example lengths",
    "",
    "Length counts Coptic letters, excluding spaces, punctuation and combining marks. Each observation is a deduplicated TLA–corpus-span pair selected through a Crum citation. Corpus spans can contain a complete verse or several clauses; their lengths differ from the printed quotation excerpt.",
    "",
    "Raw explicit Bible markers: 41,808. Unresolved or unavailable citations remain in the citation ledger and have no inferred sentence length.",
    "",
    "| Dialect | Aligned pairs | Entries | Mean letters | Median | P90 | >30 | ≤30 |",
    "| --- | --- | --- | --- | --- | --- | --- | --- |",
]
for d, s in summary.items():
    lines.append(
        f"| {d} | {s['aligned_entry_sentence_pairs']} | {s['entries']} | {s['mean_letters']:.1f} | {s['median_letters']:.1f} | {s['p90_letters']:.1f} | {s['over_30']} | {s['at_most_30']} |"
    )
lines += [
    "",
    "`entries.csv` gives every entry’s count, mean, standard deviation, quantiles, maximum and within-entry citation-order/length Spearman correlation. `examples.csv` gives individual lengths in letters, TT tokens and space-separated groups, plus within-entry Z = (length − entry mean) / entry standard deviation; constant-length groups receive Z=0. Positive order/length correlation means later citations have longer aligned spans. Citation order also reflects sense and construction grouping. These statistics describe aligned spans and provide no causal evidence of an editorial length rule. The 30-letter recommendation cap is an explicit display policy.",
]
(folder / "REPORT.md").write_text("\n".join(lines) + "\n")
print(json.dumps(summary, indent=2))
