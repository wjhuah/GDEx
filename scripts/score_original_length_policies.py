"""Evaluate length policies using the saved LightGBM held-out predictions."""

import sys, json, gzip
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.recommendation_policy import coptic_letters
from coptic_examples.ablation import expected_top_metric

root = Path(__file__).resolve().parents[1]
out = root / "reports/本地复盘/原数据八算法对照"
data = {}
for d in ["sahidic", "bohairic"]:
    for r in map(json.loads, (root / f"data/derived/model_rows_{d}.jsonl").open()):
        if r["y"] > 0:
            c = r["candidate"]
            data[d, r["tla"], r["text_key"]] = dict(
                ranks=r["ranks"],
                letters=coptic_letters(c["sent"]),
                tt_tokens=len(c["tokens"]),
                bound_groups=len(c["sent"].split()),
            )
groups = defaultdict(list)
with gzip.open(out / "留出预测.jsonl.gz", "rt") as stream:
    for line in stream:
        r = json.loads(line)
        if r["method"] != "LightGBM" or r["label"] < 0:
            continue
        source = data[r["dialect"], r["tla"], r["text_key"]]
        for p in source["ranks"]:
            groups[r["dialect"], r["fold"], r["tla"], p["source"]].append(
                dict(source, grade=p["grade"], score=r["rank_score"])
            )
rows = []
for d in ["sahidic", "bohairic"]:
    for fold in range(5):
        for unit in ["letters", "tt_tokens", "bound_groups"]:
            for mode in ["unchanged", "cutoff30", "z_penalty_0.3"]:
                for source in ["tab", "crum"]:
                    ndcg = []
                    retained = 0
                    total = 0
                    empty = 0
                    for (dd, ff, t, src), g in groups.items():
                        if (
                            dd != d
                            or ff != fold
                            or src != source
                            or len(set(r["grade"] for r in g)) < 2
                        ):
                            continue
                        x = np.array([r[unit] for r in g])
                        scores = np.array([r["score"] for r in g])
                        z = (x - x.mean()) / (x.std() or 1)
                        adjusted = (
                            scores
                            if mode == "unchanged"
                            else (
                                np.where(x <= 30, scores, -1e6)
                                if mode == "cutoff30"
                                else scores - 0.3 * np.maximum(z, 0)
                            )
                        )
                        ndcg.append(
                            expected_top_metric([r["grade"] for r in g], adjusted, 6)
                        )
                        retained += int(sum(x <= 30)) if mode == "cutoff30" else len(g)
                        total += len(g)
                        empty += int(not any(x <= 30)) if mode == "cutoff30" else 0
                    rows.append(
                        dict(
                            dialect=d,
                            fold=fold,
                            unit=unit,
                            mode=mode,
                            source=source,
                            ndcg6=float(np.mean(ndcg)) if ndcg else None,
                            groups=len(ndcg),
                            retained=retained,
                            total=total,
                            empty_groups=empty,
                        )
                    )
(out / "句长策略逐折.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
)
