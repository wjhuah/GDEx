#!/usr/bin/env python3
"""Reconcile raw Crum markers, usable links and training pairs."""

import sys, json
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config, database, write_json

cfg = Config()
counts = Counter()
usable_links = Counter()
links = defaultdict(list)
bound = defaultdict(list)
with database(cfg.work / "cache/crum_bible.sqlite3") as db:
    for oid, sid, dialect, status in db.execute(
        "SELECT occurrence_id,sentence_id,dialect,status FROM link"
    ):
        links[oid].append((sid, dialect, status))
    for oid, payload in db.execute("SELECT id,payload FROM occurrence"):
        r = json.loads(payload)
        tlas = r["tla_candidates"]
        counts["all_markers"] += 1
        if not tlas:
            reason = "unresolved_entry"
        elif len(tlas) > 1:
            reason = "ambiguous_entry"
        elif r["reference"]["kind"] != "verse":
            reason = "non_verse_reference"
        elif not links[oid]:
            reason = "no_local_link_in_applicable_dialect"
        elif not any(status == "verified_target_form" for sid, d, status in links[oid]):
            reason = "local_link_without_verified_target"
        else:
            reason = "uniquely_bound_verified_marker"
        counts[reason] += 1
        if reason == "uniquely_bound_verified_marker":
            usable_links.update(
                d for sid, d, status in links[oid] if status == "verified_target_form"
            )
        if len(tlas) == 1:
            bound[tlas[0]].append(r)
positions = {}
for tla, rows in bound.items():
    rows.sort(
        key=lambda r: (
            int("".join(c for c in r["page"] if c.isdigit()) or 0),
            r["page"].endswith("b"),
            r["file"],
            r["char_start"],
        )
    )
    for i, r in enumerate(rows, 1):
        positions[r["id"]] = dict(position=i, total=len(rows))
write_json(cfg.work / "cache/crum_source_positions.json", positions)
training = Counter()
source_map = {}
for line in (cfg.derived / "learning_examples.jsonl").open():
    r = json.loads(line)
    if r["y"] > 0 and "crum" in r["sources"]:
        training[r["dialect"]] += 1
    source_map["|".join([r["dialect"], r["tla"], r["text_key"]])] = r["ranks"]
write_json(cfg.work / "cache/source_grades.json", source_map)
write_json(
    cfg.reports / "learning/reference_counts.json",
    dict(
        exclusive_marker_counts=dict(counts),
        verified_links_by_dialect=dict(
            Counter(
                d
                for rs in links.values()
                for sid, d, status in rs
                if status == "verified_target_form"
            )
        ),
        uniquely_bound_verified_links_by_dialect=dict(usable_links),
        training_crum_pairs=dict(training),
    ),
)
print(dict(counts))
print("training", dict(training))
