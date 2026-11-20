pass

import sys, json, re, csv
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config, write_json
from coptic_examples.recommendation_policy import coptic_letters

cfg = Config()
folder = cfg.reports / 'local_2218a127/local_77165ac0'
folder.mkdir(parents=True, exist_ok=True)
source = cfg.reports / 'local_2218a127/local_40c6713e'
rows = list(map(json.loads, (source / '').open()))
print(rows[0].keys())

groups = defaultdict(list)
for r in rows:
    text = r.get("text", r.get("sent", ""))
    groups[r["tla"]].append(
        dict(
            tla=r["tla"],
            sentence_id=r["sentence_id"],
            letters=coptic_letters(text),
            bound_groups=len(text.split()),
            text=text,
        )
    )
entry = []
samples = []
for tla, rr in groups.items():
    lengths = np.array([r["letters"] for r in rr])
    mean = float(lengths.mean())
    sd = float(lengths.std())
    for r in rr:
        r["within_entry_z"] = (r["letters"] - mean) / sd if sd else 0
        samples.append(r)
    entry.append(
        dict(
            tla=tla,
            n=len(rr),
            mean=mean,
            std=sd,
            median=float(np.median(lengths)),
            p90=float(np.quantile(lengths, 0.9)),
        )
    )
write_json(
    folder / '',
    dict(
        examples=samples,
        entries=entry,
        summary=dict(
            pairs=len(samples),
            mean_letters=float(np.mean([r["letters"] for r in samples])),
            over30=sum(r["letters"] > 30 for r in samples),
        ),
    ),
)
rare = []
for path in sorted((cfg.source / "Crum_plain_text").glob("*.txt")):
    text = path.read_text()
    for m in re.finditer(r"\brare(?:ly)?\b", text, re.I):
        before = text[max(0, m.start() - 120) : m.start()]
        after = text[m.end() : m.end() + 120]
        
        rare.append(
            dict(
                file=path.name,
                offset=m.start(),
                context=before + m[0] + after,
                nearby_S=bool(re.search("⟪S⟫", before[-60:] + after[:60])),
            )
        )
write_json(folder / '', rare)
write_json(
    folder / '',
    dict(
        mentions=len(rare),
        files=len(set(r["file"] for r in rare)),
        S_within_60_chars=sum(r["nearby_S"] for r in rare),
        scope='',
    ),
)
