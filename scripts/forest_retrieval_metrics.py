#!/usr/bin/env python3
"""Evaluate filtered output on the same judged candidate pools."""

import sys, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config, write_json
from coptic_examples.recommendation_policy import coptic_letters


def metric(groups, k=6):
    ndcg = []
    p3 = []
    for values in groups.values():
        if not any(g > 0 for g, s, ok in values):
            continue
        ideal = sorted([2**g - 1 for g, s, ok in values], reverse=True)[:k]
        denominator = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
        retained = [(g, s) for g, s, ok in values if ok]
        tiers = defaultdict(list)
        for g, s in retained:
            tiers[s].append(g)
        numerator = 0.0
        hits = 0.0
        offset = 0
        for score in sorted(tiers, reverse=True):
            gains = tiers[score]
            mean = float(np.mean([2**g - 1 for g in gains]))
            positive = float(np.mean([g > 0 for g in gains]))
            for i in range(offset, min(offset + len(gains), k)):
                numerator += mean / math.log2(i + 2)
            for i in range(offset, min(offset + len(gains), 3)):
                hits += positive
            offset += len(gains)
        ndcg.append(numerator / denominator if denominator else 0)
        p3.append(hits / 3)
    return dict(
        groups=len(ndcg),
        ndcg_at_6=float(np.mean(ndcg)) if ndcg else None,
        judged_precision_at_3=float(np.mean(p3)) if p3 else None,
    )


def main():
    cfg = Config()
    path = cfg.reports / "forest/comparison.json"
    report = json.loads(path.read_text())
    for result in report["results"]:
        d = result["dialect"]
        rows = json.loads((cfg.reports / f"forest/oof_{d}.json").read_text())
        length = {}
        for line in (cfg.derived / f"model_rows_{d}.jsonl").open():
            r = json.loads(line)
            length[(r["tla"], r["candidate"]["id"])] = coptic_letters(
                r["candidate"]["sent"]
            )
        output = {}
        for name in ["lgb", "rf"]:
            threshold = result["classification"][name]["development_threshold"][
                "threshold"
            ]
            for capped in [False, True]:
                groups = defaultdict(list)
                for r in rows:
                    grades = {s["source"]: s["grade"] for s in r["ranks"]}
                    for source in ["tab", "crum"]:
                        if r["y"] > 0 and source not in grades:
                            continue
                        grade = grades.get(source, 0) if r["y"] > 0 else 0
                        accepted = r[name] > threshold and (
                            not capped or length[(r["tla"], r["id"])] <= 30
                        )
                        groups[(r["tla"], source)].append((grade, r["rank"], accepted))
                output[name + ("_cap30" if capped else "")] = {
                    source: metric({k: v for k, v in groups.items() if k[1] == source})
                    for source in ["tab", "crum"]
                }
        result["filtered_judged_candidate_ranking"] = output
    report["filtered_ranking_protocol"] = (
        "Same original judged pool for both algorithms. Negatives gain=0; positive source preference grades retained. Missing rejected candidates earn zero DCG against the full-pool ideal. Ties use expected gain. P@3 divides by three including empty slots. Development-selected thresholds; no token-sense head or dialect restriction. cap30 additionally applies the display length policy."
    )
    write_json(path, report)
    lines = [
        "\n## Filtered judged-pool ranking\n",
        report["filtered_ranking_protocol"],
        "",
        "| Dialect | Classifier / filter | Source | NDCG@6 | Judged P@3 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in report["results"]:
        for model, sources in r["filtered_judged_candidate_ranking"].items():
            for source, m in sources.items():
                lines.append(
                    f"| {r['dialect']} | {model} | {source} | {m['ndcg_at_6']:.4f} | {m['judged_precision_at_3']:.4f} |"
                )
    with (cfg.reports / "forest/REPORT.md").open("a") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
