pass

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config
from coptic_examples.dataset import text_key

cfg = Config()
folder = cfg.reports / '' / (sys.argv[1] if len(sys.argv) > 1 else '')
known = {}
for line in (cfg.derived / "learning_examples.jsonl").open():
    r = json.loads(line)
    known[(r["tla"], r["dialect"], r["text_key"])] = r["y"]
unique = {}
seen = Counter()
for line in (folder / '').open():
    r = json.loads(line)
    key = (r["tla"], r["dialect"], text_key({"sent": r["text"]}))
    state = (
        ''
        if known.get(key, 0) > 0
        else '' if known.get(key, 0) < 0 else ''
    )
    seen[state] += 1
    item = unique.setdefault(key, dict(r, training_status=state))
    item["citation_ids"] = sorted(set(item["citation_ids"]) | set(r["citation_ids"]))
with (folder / '').open("w") as f:
    for r in unique.values():
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
counts = Counter(r["training_status"] for r in unique.values())
summary = json.loads((folder / '').read_text())
summary["deduplicated_recovered_pairs"] = counts
summary["deduplicated_pair_total"] = len(unique)
(folder / '').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
)
groups = defaultdict(Counter)
for line in (folder / '').open():
    r = json.loads(line)
    groups[r["old_status"]][r["status"]] += 1
lines = [
    '',
    "",
    '',
    "",
    '',
    "",
    '',
    "|---|---|---|",
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    '',
    "",
    '',
    "",
    '',
    "|---|---:|",
]
lines += [f"| {k} | {v:,} |" for k, v in summary["statuses"].items()]
lines += [
    "",
    ''
    + str(sum(summary["recovered_occurrences_by_previous_status"].values()))
    + '',
    "",
    '',
    "|---|---:|",
]
lines += [f"| {k} | {v:,} |" for k, v in counts.items()]
lines += [
    "",
    '',
]
(folder / '').write_text("\n".join(lines) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
