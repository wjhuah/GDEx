pass
import csv
import json
from collections import Counter
from pathlib import Path

folder = Path(__file__).resolve().parents[1] / "reports" / ''

def read(name):
    with (folder / name).open() as stream:
        for line in stream:
            yield json.loads(line)

cohort = {r["id"]: r for r in read('')}
readings = {}
for name in ['', '']:
    for r in read(name):
        assert r["id"] not in readings
        readings[r["id"]] = r["direct_reading"]
assert len(cohort) == 29998 and len(readings) == 31
assert set(readings) <= set(cohort)
counts = Counter()
dest = folder / ''
with dest.with_suffix(".jsonl").open("w") as detail, dest.with_suffix(".tsv").open("w") as table:
    writer = csv.writer(table, delimiter="\t")
    writer.writerow(['', '', '', '', '', '', '', '', ''])
    for oid, record in cohort.items():
        reading = readings.get(oid)
        if reading:
            status = ''
            reason = reading["reason"]
        elif not record["versions"]:
            status = ''
            reason = ''
        else:
            status = ''
            reason = record["previous_reason"]
        counts[status] += 1
        row = dict(record, review_status=status, direct_reading=reading)
        detail.write(json.dumps(row, ensure_ascii=False) + "\n")
        writer.writerow([oid, record["reference"], record["page"], record["heading"], status, reason,
                         reading.get("tla", "") if reading else "",
                         " ".join(reading.get("tokens", [reading.get("token", "")])) if reading else "",
                         len(record["versions"])])
assert sum(counts.values()) == 29998
report = [
    '', "",
    '',
    '', "",
    '', "| --- | ---: |",
    *[f"| {k} | {v:,} |" for k, v in counts.items()], "",
    '',
    '',
    '',
]
(folder / '').write_text("\n".join(report) + "\n")
print(json.dumps(counts, ensure_ascii=False))
