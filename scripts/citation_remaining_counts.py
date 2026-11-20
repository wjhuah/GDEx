pass

import csv
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[1]
folder = (
    root / 'reports/local_2218a127' / (sys.argv[1] if len(sys.argv) > 1 else '')
)
statuses = Counter()
remaining = Counter()
causes = Counter()
seen = set()
scoped = (folder / '').exists()
original = {}
if scoped:
    for line in (root / "data/derived/complete_citations/citations.jsonl").open():
        record = json.loads(line)
        original[record["id"]] = record
with (folder / '').open("w") as output:
    writer = csv.writer(output, delimiter="\t")
    writer.writerow(['', '', '', '', '', ''])
    for line in (folder / '').open():
        r = json.loads(line)
        assert r["id"] not in seen
        seen.add(r["id"])
        if scoped and "S" not in r["original_dialect_codes"]:
            statuses[''] += 1
            continue
        existing = (
            any(p["dialect"] == "sahidic" for p in original[r["id"]]["training_pairs"])
            if scoped
            else r["old_status"] == "training_pair_available"
        )
        if existing:
            statuses[''] += 1
        elif r["accepted"]:
            statuses[''] += 1
            causes[r["binding_basis"]] += 1
        else:
            statuses[''] += 1
            remaining[r["status"]] += 1
            writer.writerow(
                [
                    r["id"],
                    r["reference"],
                    r["file"],
                    r["page"],
                    ",".join(r["proposed_tlas"]),
                    r["status"],
                ]
            )
assert len(seen) == sum(statuses.values()) == 41808
result = dict(
    total_occurrences=41808,
    exclusive_statuses=statuses,
    pending_reasons=remaining,
    automatic_binding_evidence=causes,
)
(folder / '').write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n"
)
new = json.loads((folder / '').read_text())["deduplicated_recovered_pairs"]
lines = [
    '',
    "",
    '',
    '',
    "",
    '',
    "|---|---:|",
]
lines += [f"| {k} | {n:,} |" for k, n in statuses.items()]
lines += ["", '', "|---|---:|"]
lines += [f"| {k} | {n:,} |" for k, n in remaining.items()]
lines += ["", '', "|---|---:|"]
lines += [f"| {k} | {n:,} |" for k, n in new.items()]
lines += [
    "",
    '',
    '',
    '',
]
(folder / '').write_text("\n".join(lines) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
