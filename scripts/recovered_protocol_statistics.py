import sys, json, re, sqlite3
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from scipy.stats import spearmanr

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from coptic_examples.recommendation_policy import coptic_letters

folder = root / "reports/本地复盘/新增数据模型比较"
db = sqlite3.connect(folder / "实验数据.sqlite3")
groups = defaultdict(list)
positions = {
    r["id"]: (
        int("".join(filter(str.isdigit, r["page"])) or 0),
        r["page"].endswith("b"),
        r["file"],
        r["char_start"],
    )
    for r in map(
        json.loads, (root / "data/derived/complete_citations/citations.jsonl").open()
    )
}
for (payload,) in db.execute("select payload from example"):
    r = json.loads(payload)
    if r["y"] < 0 or not any(s in r["sources"] for s in ["crum", "crum_automatic"]):
        continue
    c = r["candidate"]
    ids = r.get("citation_ids", []) + [
        v.get("occurrence_id") for v in r.get("crum_refs", [])
    ]
    pos = min((positions[i] for i in ids if i in positions), default=None)
    groups[(r["dialect"], r["tla"])].append(
        dict(
            dialect=r["dialect"],
            tla=r["tla"],
            id=c["id"],
            letters=coptic_letters(c["sent"]),
            tt_tokens=len(c["tokens"]),
            bound_groups=len(c["sent"].split()),
            order=pos,
        )
    )
examples = []
entries = []
for (d, t), rr in groups.items():
    item = dict(dialect=d, tla=t, n=len(rr))
    rr.sort(key=lambda r: r["order"] or (99999, False, "", 0))
    for unit in ["letters", "tt_tokens", "bound_groups"]:
        x = np.array([r[unit] for r in rr], dtype=float)
        mean = float(x.mean())
        sd = float(x.std())
        item[unit] = dict(
            mean=mean,
            sd=sd,
            median=float(np.median(x)),
            p90=float(np.quantile(x, 0.9)),
            over30=int(sum(x > 30)),
            order_rho=(
                float(spearmanr(range(len(x)), x).statistic)
                if len(x) >= 3 and sd > 0
                else None
            ),
        )
        for r in rr:
            r[unit + "_z"] = (r[unit] - mean) / sd if sd else 0
    examples += rr
    entries.append(item)
(folder / "合并数据逐词条句长.json").write_text(
    json.dumps(entries, ensure_ascii=False, indent=2)
)
(folder / "合并数据逐句句长.jsonl").write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in examples)
)
summary = {}
for d in ["sahidic", "bohairic"]:
    rr = [r for r in examples if r["dialect"] == d]
    summary[d] = {"pairs": len(rr)}
    for unit in ["letters", "tt_tokens", "bound_groups"]:
        x = [r[unit] for r in rr]
        rho = [
            r[unit]["order_rho"]
            for r in entries
            if r["dialect"] == d and r[unit]["order_rho"] is not None
        ]
        summary[d][unit] = dict(
            mean=float(np.mean(x)),
            median=float(np.median(x)),
            over30=sum(v > 30 for v in x),
            kept30=sum(v <= 30 for v in x),
            median_within_entry_rho=float(np.median(rho)) if rho else None,
        )
(folder / "合并句长摘要.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2)
)
rare = []
for path in sorted((root.parent / "Crum_plain_text").glob("*.txt")):
    text = path.read_text()
    for match in re.finditer(r"⟪S⟫\s*\(rare\)", text):
        rare.append(
            dict(
                file=path.name,
                offset=match.start(),
                source_paragraph=text[
                    max(0, text.rfind("¶", 0, match.start())) : (
                        text.find("\n¶", match.end())
                        if "\n¶" in text[match.end() :]
                        else len(text)
                    )
                ],
            )
        )
(folder / "明确S少见标记.json").write_text(
    json.dumps(rare, ensure_ascii=False, indent=2)
)
print(summary)
print("explicit S rare", len(rare), "files", len(set(r["file"] for r in rare)))
