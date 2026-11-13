"""Fixed-fold preference comparison for explicit length policies."""

import sys, json, sqlite3
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config, write_json
from coptic_examples.learning import fit_score, predict
from coptic_examples.recommendation_policy import coptic_letters
from coptic_examples.ablation import expected_top_metric

cfg = Config()
folder = cfg.reports / "本地复盘/新增数据模型比较"
db = sqlite3.connect(folder / "实验数据.sqlite3")
rows = [
    json.loads(v[0])
    for v in db.execute(
        "select payload from example where cohort='original' and dialect='sahidic'"
    )
]
db.close()
results = []
for fold in range(5):
    val = [r for r in rows if r["fold"] == fold and r["y"] > 0]
    held = {r["text_key"] for r in rows if r["fold"] == fold}
    tr = [r for r in rows if r["fold"] != fold and r["text_key"] not in held]
    score = predict(fit_score(tr, False), val)
    for unit in ["letters", "tt_tokens", "bound_groups"]:
        groups = defaultdict(list)
        for r, s in zip(val, score):
            c = r["candidate"]
            n = (
                coptic_letters(c["sent"])
                if unit == "letters"
                else len(c["tokens"]) if unit == "tt_tokens" else len(c["sent"].split())
            )
            for p in r["ranks"]:
                groups[(r["tla"], p["source"])].append((p["grade"], float(s), n))
        for mode in ["unchanged", "cutoff30", "z_penalty_0.3"]:
            for source in ["tab", "crum"]:
                values = []
                kept = 0
                total = 0
                for (t, src), g in groups.items():
                    if src != source or len(set(a for a, b, c in g)) < 2:
                        continue
                    y = [a for a, b, c in g]
                    s = np.array([b for a, b, c in g])
                    length = np.array([c for a, b, c in g])
                    z = (length - length.mean()) / (length.std() or 1)
                    adjusted = (
                        s
                        if mode == "unchanged"
                        else (
                            np.where(length <= 30, s, -1e6)
                            if mode == "cutoff30"
                            else s - 0.3 * np.maximum(z, 0)
                        )
                    )
                    values.append(expected_top_metric(y, adjusted, 6))
                    kept += int(sum(length <= 30)) if mode == "cutoff30" else len(g)
                    total += len(g)
                results.append(
                    dict(
                        fold=fold,
                        unit=unit,
                        mode=mode,
                        source=source,
                        groups=len(values),
                        ndcg6=float(np.mean(values)) if values else None,
                        retained=kept,
                        total=total,
                    )
                )
write_json(folder / "句长策略对照.json", results)
