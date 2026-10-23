#!/usr/bin/env python3
"""Add same-lemma Crum hard negatives and a capped number of random negatives."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from retrieval import normalize_lemma


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "training_v15_merged.tsv"
DEFAULT_OUTPUT = Path(__file__).with_name("training_with_negatives.tsv")


def _is_crum(row: dict[str, str]) -> bool:
    source = (row.get("source") or "").casefold()
    return (
        source == "both"
        or ("crum" in source and not source.startswith(("hard_negative:", "random_negative:")))
    )


def _negative(target: dict[str, str], example: dict[str, str], kind: str) -> dict[str, str]:
    row = target.copy()
    for field in ("sent", "translation", "ref"):
        if field in row:
            row[field] = example.get(field, "")
    row["label"] = "0.0"
    row["forbidden"] = "0"
    row["source"] = f"{kind}:{example.get('source', '')}"
    if "crum_rank" in row:
        row["crum_rank"] = ""
    if "crum_sense_idx" in row:
        row["crum_sense_idx"] = ""
    if "tab_priority" in row:
        row["tab_priority"] = ""
    if "tab_notes" in row:
        row["tab_notes"] = f"Generated {kind.replace('_', ' ')}."
    return row


def add_negative_examples(
    rows: list[dict[str, str]],
    random_ratio: float = 0.10,
    seed: int = 42,
    include_hard: bool = True,
    include_random: bool = True,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    if not 0.0 <= random_ratio <= 0.10:
        raise ValueError("random_ratio must be between 0 and 0.10.")
    by_tla: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_lemma: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        tla = row.get("TLA", "").strip()
        lemma = normalize_lemma(row.get("lemma", ""))
        if tla:
            by_tla[tla].append(row)
        if lemma and tla and _is_crum(row):
            by_lemma[lemma].append(row)

    generated: list[dict[str, str]] = []
    hard_count = 0
    random_count = 0
    rng = random.Random(seed)
    for tla in sorted(by_tla):
        target_rows = by_tla[tla]
        target_lemma = normalize_lemma(target_rows[0].get("lemma", ""))
        target_examples = [row for row in target_rows if _is_crum(row)]
        if not target_examples:
            continue
        existing = {
            (row.get("sent", ""), row.get("translation", ""))
            for row in target_rows
        }
        if include_hard:
            ambiguous_examples = [
                row for row in by_lemma.get(target_lemma, [])
                if row.get("TLA", "").strip() != tla
            ]
            for target in target_examples:
                for example in ambiguous_examples:
                    key = (example.get("sent", ""), example.get("translation", ""))
                    if key in existing:
                        continue
                    generated.append(_negative(target, example, "hard_negative"))
                    existing.add(key)
                    hard_count += 1

        base_examples = len(target_examples)
        random_cap = int(base_examples * random_ratio) if include_random else 0
        if not random_cap:
            continue
        random_pool = [
            row for other_tla, examples in sorted(by_tla.items())
            if other_tla != tla
            for row in examples
            if _is_crum(row) and normalize_lemma(row.get("lemma", "")) != target_lemma
            and (row.get("sent", ""), row.get("translation", "")) not in existing
        ]
        rng.shuffle(random_pool)
        for example in random_pool[:random_cap]:
            target = target_examples[0]
            generated.append(_negative(target, example, "random_negative"))
            existing.add((example.get("sent", ""), example.get("translation", "")))
            random_count += 1

    return rows + generated, {
        "original_rows": len(rows),
        "hard_negatives": hard_count,
        "random_negatives": random_count,
        "total_rows": len(rows) + len(generated),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--random-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    negative_type = parser.add_mutually_exclusive_group()
    negative_type.add_argument(
        "--hard-only", action="store_true",
        help="Generate same-lemma hard negatives only.",
    )
    negative_type.add_argument(
        "--random-only", action="store_true",
        help="Generate random negatives only.",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input file not found: {args.input}")
    with args.input.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not reader.fieldnames:
            parser.error("input TSV has no header.")
        fieldnames = reader.fieldnames
        rows = list(reader)
    output, stats = add_negative_examples(
        rows,
        args.random_ratio,
        args.seed,
        include_hard=not args.random_only,
        include_random=not args.hard_only,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(output)
    print(
        f"Wrote {stats['total_rows']} rows to {args.output} "
        f"({stats['hard_negatives']} hard, {stats['random_negatives']} random)."
    )


if __name__ == "__main__":
    main()
