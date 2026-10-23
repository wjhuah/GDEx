#!/usr/bin/env python3
"""Evaluate Crum4 ranking on held-out-by-group Crum3 training candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from retrieval import (
    DEFAULT_LEXICON,
    DEFAULT_MODEL,
    EmbeddingCache,
    Lexicon,
    RetrievalEngine,
    normalize_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "training_v15_with_sm_bge.tsv"
DEFAULT_OUTPUT = Path(__file__).with_name("evaluation_crum4.json")


def label_to_grade(label_int: int) -> int:
    label_int = max(0, min(30, int(label_int)))
    if label_int >= 30:
        return 3
    if label_int >= 20:
        return 2
    if label_int >= 10:
        return 1
    return 0


def ndcg_at_k(labels: list[int], scores: list[float], k: int = 6) -> tuple[float, float]:
    """Match train_v15_sm.py's discrete-gain old and raw-grade linear metrics."""
    cutoff = min(k, len(labels))
    order = sorted(range(len(labels)), key=lambda index: -scores[index])[:cutoff]
    ideal = sorted(labels, reverse=True)[:cutoff]
    discounts = [math.log2(rank + 2) for rank in range(cutoff)]
    old_dcg = sum(
        ((2 ** label_to_grade(labels[index])) - 1) / discounts[rank]
        for rank, index in enumerate(order)
    )
    old_idcg = sum(
        ((2 ** label_to_grade(label)) - 1) / discounts[rank]
        for rank, label in enumerate(ideal)
    )
    linear_dcg = sum(labels[index] / discounts[rank] for rank, index in enumerate(order))
    linear_idcg = sum(label / discounts[rank] for rank, label in enumerate(ideal))
    old = old_dcg / old_idcg if old_idcg else 0.0
    linear = linear_dcg / linear_idcg if linear_idcg else 0.0
    return old, linear


def load_groups(path: Path) -> dict[str, list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation TSV not found: {path}")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if not row.get("TLA") or row.get("forbidden", "0") == "1":
                continue
            try:
                float(row["label"])
            except (KeyError, ValueError):
                continue
            groups[row["TLA"]].append(row)
    return {tla: rows for tla, rows in groups.items() if len(rows) >= 2}


def evaluate(
    input_path: Path,
    lexicon_path: Path = DEFAULT_LEXICON,
    model_name: str = DEFAULT_MODEL,
    sample_size: int = 100,
    seed: int = 42,
) -> dict:
    if sample_size < 100:
        raise ValueError("sample_size must be at least 100 TLA groups.")
    groups = load_groups(input_path)
    if len(groups) < sample_size:
        raise ValueError(
            f"Need at least {sample_size} TLA groups with two candidates; found {len(groups)}."
        )
    rng = random.Random(seed)
    selected_tlas = sorted(rng.sample(sorted(groups), sample_size))
    selected = {tla: groups[tla] for tla in selected_tlas}
    lexicon = Lexicon.load(lexicon_path)

    glosses: dict[str, str] = {}
    candidates_by_tla: dict[str, list[dict]] = {}
    all_sentences = []
    for tla, rows in selected.items():
        entry = lexicon.by_tla.get(tla)
        gloss = next(
            (normalize_text(row.get("crum_gloss", "")) for row in rows
             if normalize_text(row.get("crum_gloss", ""))),
            "",
        )
        if not gloss and entry:
            gloss = entry.gloss
        if not gloss and entry:
            gloss = entry.lemma
        if not gloss:
            raise ValueError(f"No definition/gloss available for sampled TLA {tla}.")
        glosses[tla] = gloss
        candidates = []
        for index, row in enumerate(rows):
            try:
                frequency = float(row.get("mean_word_frequency", 0) or 0)
            except ValueError:
                frequency = 0.0
            candidate = {
                "sent": normalize_text(row.get("sent", "")),
                "translation": normalize_text(row.get("translation", "")),
                "source": row.get("source", ""),
                "ref": row.get("ref", ""),
                "mean_frequency": frequency,
                "label_int": max(0, min(30, int(round(float(row["label"]) * 10)))),
                "row_index": index,
            }
            candidates.append(candidate)
            all_sentences.append(candidate["translation"] or candidate["sent"])
        candidates_by_tla[tla] = candidates

    cache = EmbeddingCache(model_name=model_name)
    definition_vectors = cache.get_many("definitions", list(glosses.values()))
    sentence_vectors = cache.get_many("sentences", all_sentences)
    engine = RetrievalEngine(embeddings=cache)
    group_metrics = []
    for tla, candidates in candidates_by_tla.items():
        ranked = engine.score_candidates_preembedded(
            candidates,
            glosses[tla],
            sentence_vectors,
            definition_vectors[glosses[tla]],
        )
        labels = [candidate["label_int"] for candidate in candidates]
        scores_by_index = {
            candidate["row_index"]: candidate["score"] for candidate in ranked
        }
        scores = [scores_by_index[index] for index in range(len(candidates))]
        ndcg_old, ndcg_linear = ndcg_at_k(labels, scores)
        group_metrics.append({
            "TLA": tla,
            "lemma": (
                lexicon.by_tla[tla].lemma
                if tla in lexicon.by_tla
                else selected[tla][0].get("lemma", "")
            ),
            "gloss": glosses[tla],
            "candidate_count": len(candidates),
            "ndcg_at_6_old": ndcg_old,
            "ndcg_at_6_linear": ndcg_linear,
        })
    group_metrics.sort(key=lambda item: item["TLA"])
    return {
        "protocol": {
            "input": str(input_path),
            "sample_size": sample_size,
            "eligible_groups": len(groups),
            "seed": seed,
            "k": 6,
            "old_gain": "2^rel - 1, rel bins: >=30->3, >=20->2, >=10->1, else 0",
            "linear_gain": "raw clipped label_int in [0, 30], matching train_v15_sm.py",
            "model": model_name,
            "ranking": "0.55*cosine + 0.30*ROUGE-L + 0.15*frequency",
        },
        "mean_ndcg_at_6_old": sum(row["ndcg_at_6_old"] for row in group_metrics) / len(group_metrics),
        "mean_ndcg_at_6_linear": sum(row["ndcg_at_6_linear"] for row in group_metrics) / len(group_metrics),
        "groups": group_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = evaluate(
        args.input, args.lexicon, args.model, args.sample_size, args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Eligible TLA groups: {report['protocol']['eligible_groups']}")
    print(f"Evaluated TLA groups: {report['protocol']['sample_size']}")
    print(f"NDCG@6 (old): {report['mean_ndcg_at_6_old']:.4f}")
    print(f"NDCG@6 (linear): {report['mean_ndcg_at_6_linear']:.4f}")
    print(f"Detailed output: {args.output}")


if __name__ == "__main__":
    main()
