#!/usr/bin/env python3
"""Run a reproducible, corpus-wide BGE review cohort and compare local CDO examples."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from retrieval import (
    DEFAULT_LEXICON,
    DEFAULT_MODEL,
    EmbeddingCache,
    Lexicon,
    RetrievalEngine,
    _coarse_score,
    normalize_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CDO_DATA_DIR = PROJECT_ROOT.parent / "dictionary"
DEFAULT_OUTPUT = Path(__file__).with_name("TLA_REVIEW.md")
KNOWN_HOMONYM_PAIRS = (
    ("C998", "C992"),
    ("C19", "C20"),
    ("C7376", "C7377"),
    ("C5972", "C5995"),
    ("C3092", "C3093"),
)
KNOWN_CASES = {"C998", "C992", "C6"}
POS_CATEGORIES = ("NOUN", "VERB", "ADJ", "OTHER")


def choose_cohort(lexicon: Lexicon, frequencies: dict[tuple[str, str], int]) -> tuple[dict[str, set[str]], dict[str, int]]:
    entries = [
        (entry, frequencies.get((entry.lemma, entry.pos or ""), 0))
        for entry in lexicon.by_tla.values()
        if entry.pos
    ]
    positive = [(entry, count) for entry, count in entries if count > 0]
    high = sorted(positive, key=lambda pair: (-pair[1], pair[0].tla))[:10]
    low_pool = sorted(positive, key=lambda pair: (pair[1], pair[0].tla))
    low_30_size = max(10, math.ceil(len(low_pool) * 0.30))
    low = low_pool[:low_30_size][:10]

    roles: dict[str, set[str]] = defaultdict(set)
    for entry, _ in high:
        roles[entry.tla].add("high-frequency")
    for entry, _ in low:
        roles[entry.tla].add("low-frequency")

    for pair_index, pair in enumerate(KNOWN_HOMONYM_PAIRS, 1):
        for tla in pair:
            roles[tla].add(f"homograph-{pair_index}")
    for tla in KNOWN_CASES:
        roles[tla].add("known-case")

    for pos in POS_CATEGORIES:
        members = [
            (entry, frequencies.get((entry.lemma, entry.pos or ""), 0))
            for entry, _ in entries
            if (entry.pos == pos if pos != "OTHER" else entry.pos not in {"NOUN", "VERB", "ADJ"})
        ]
        present = sorted(
            (entry for entry, _ in members if entry.tla in roles),
            key=lambda entry: entry.tla,
        )
        chosen = present[:5]
        if len(chosen) < 5:
            ranked = sorted(
                (pair for pair in members if pair[0] not in chosen),
                key=lambda pair: (-pair[1], pair[0].tla),
            )
            chosen.extend(entry for entry, _ in ranked[:5 - len(chosen)])
        for entry in chosen:
            roles[entry.tla].add(f"POS-{pos}")

    frequencies_by_tla = {
        tla: frequencies.get((entry.lemma, entry.pos or ""), 0)
        for tla, entry in lexicon.by_tla.items()
    }
    return dict(sorted(roles.items())), frequencies_by_tla


def load_cdo_examples(tlas: set[str]) -> dict[str, list[dict[str, str]]]:
    output: dict[str, list[dict[str, str]]] = defaultdict(list)
    for filename in ("examples_sahidic.tab", "examples_bohairic.tab"):
        path = CDO_DATA_DIR / filename
        if not path.is_file():
            continue
        dialect = "Sahidic" if "sahidic" in filename else "Bohairic"
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.reader(stream, delimiter="\t"):
                if len(row) < 8 or row[1] not in tlas:
                    continue
                output[row[1]].append({
                    "lemma": row[0],
                    "pos": row[2],
                    "sent": row[3].replace("**", ""),
                    "translation": row[4],
                    "title": row[5],
                    "urn": row[6],
                    "priority": row[7],
                    "dialect": row[8] if len(row) > 8 else dialect,
                })
    return output


def _display_grade(candidate: dict | None, candidate_count: int) -> tuple[str, str]:
    if not candidate_count:
        return "差", "没有同 lemma+POS 的语料命中，属于语料标注/覆盖缺口。"
    if candidate is None or not candidate["translation"]:
        return "差", "top-1 缺少译文，当前无法可靠判断义项适配。"
    if candidate["rouge_l"] >= 0.20:
        return "好", f"定义与译文有较强内容词重合（ROUGE-L={candidate['rouge_l']:.2f}）。"
    if candidate["rouge_l"] > 0:
        return "中", f"有少量定义词重合（ROUGE-L={candidate['rouge_l']:.2f}），建议人工核义。"
    return "差", "top-1 译文与定义没有内容词重合，需检查语义排序与译文质量。"


def _md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def run_review(model_name: str = DEFAULT_MODEL, limit: int = 100) -> tuple[str, dict]:
    lexicon = Lexicon.load(DEFAULT_LEXICON)
    engine = RetrievalEngine(embeddings=EmbeddingCache(model_name=model_name))
    counts = engine.index.lemma_pos_counts()
    roles, frequencies = choose_cohort(lexicon, counts)
    if len(roles) < 30:
        raise RuntimeError(f"Review cohort has only {len(roles)} unique TLAs; at least 30 are required.")
    tlas = list(roles)
    cdo = load_cdo_examples(set(tlas))

    candidate_sets = {}
    definition_by_tla = {}
    sentence_texts = []
    for tla in tlas:
        entry = lexicon.by_tla.get(tla)
        if entry is None or entry.pos is None:
            candidate_sets[tla] = []
            continue
        definition = entry.gloss or entry.lemma
        definition_by_tla[tla] = definition
        candidates = engine.index.search(entry.lemma, entry.pos)
        unique = {}
        for candidate in candidates:
            key = (
                "".join(normalize_text(candidate["sent"]).casefold().split()),
                normalize_text(candidate["translation"]).casefold(),
            )
            unique.setdefault(key, candidate)
        candidates = list(unique.values())
        candidates.sort(
            key=lambda candidate: (
                -_coarse_score(candidate, definition),
                candidate["source"], candidate["ref"], candidate["sent"],
            )
        )
        candidate_sets[tla] = candidates[:limit]
        sentence_texts.extend(
            item["translation"] or item["sent"] for item in candidate_sets[tla]
        )

    cache = engine.embeddings
    definition_vectors = cache.get_many("definitions", list(definition_by_tla.values()))
    sentence_vectors = cache.get_many("sentences", sentence_texts)
    reviews = []
    detail_rows = []
    for tla in tlas:
        entry = lexicon.by_tla.get(tla)
        candidates = candidate_sets[tla]
        if entry is not None and candidates:
            ranked = engine.score_candidates_preembedded(
                candidates,
                definition_by_tla[tla],
                sentence_vectors,
                definition_vectors[definition_by_tla[tla]],
            )
        else:
            ranked = []
        top = ranked[0] if ranked else None
        grade, reason = _display_grade(top, len(candidates))
        cdo_rows = cdo.get(tla, [])
        reviews.append({
            "tla": tla,
            "lemma": entry.lemma if entry else "",
            "pos": entry.pos if entry else "",
            "gloss": entry.gloss if entry else "",
            "frequency": frequencies.get(tla, 0),
            "candidate_count": len(engine.index.search(entry.lemma, entry.pos))
            if entry and entry.pos else 0,
            "top": top,
            "grade": grade,
            "reason": reason,
            "cdo_rows": cdo_rows,
            "roles": sorted(roles[tla]),
            "ranked": ranked,
        })
        if ranked:
            for rank, result in enumerate(ranked[:10], 1):
                detail_rows.append(
                    f"| {tla} | {rank} | {result['score']:.3f} | "
                    f"{result['cosine']:.3f} | {result['rouge_l']:.3f} | "
                    f"{result['translation'] or '（无译文）'} | "
                    f"`{result['source']}` · {result['ref']} | "
                    f"{'CDO 词条例句见左侧' if cdo_rows else 'CDO 无数据'} |"
                )
        else:
            detail_rows.append(f"| {tla} | — | — | — | — | （无候选） | — | {'有 CDO 数据' if cdo_rows else 'CDO 无数据'} |")

    top_rows = []
    cdo_comparison_rows = []
    for review in reviews:
        top_translation = (
            review["top"]["translation"].replace("|", "\\|")
            if review["top"] and review["top"]["translation"] else "（无译文/候选）"
        )
        top_rows.append(
            f"| {review['tla']} | {review['lemma']} | {review['pos']} | "
            f"{review['gloss'].replace('|', ';')} | {review['candidate_count']} | "
            f"{top_translation} | {review['grade']} | {review['reason']} | "
            f"{'CDO ' + str(len(review['cdo_rows'])) + ' 条' if review['cdo_rows'] else 'CDO 无数据'} | "
            f"{', '.join(review['roles'])} |"
        )
        for example in review["cdo_rows"]:
            cdo_comparison_rows.append(
                f"| {review['tla']} | {review['lemma']} | {review['pos']} | "
                f"{example['dialect']} | "
                f"{_md_cell(example['sent'])} | "
                f"{_md_cell(example['translation'])} | "
                f"`{_md_cell(example['title'])} · {_md_cell(example['urn'])}` | "
                f"{top_translation} | "
                f"{_md_cell(review['top']['ref']) if review['top'] else '（无候选）'} |"
            )
    summary = (
        f"本轮共测 {len(reviews)} 个 TLA；候选非零 {sum(bool(item['candidate_count']) for item in reviews)} 个，"
        f"CDO 本地 curated examples 有数据 {sum(bool(item['cdo_rows']) for item in reviews)} 个。"
        f"自动初筛评价：好 {sum(item['grade']=='好' for item in reviews)}，"
        f"中 {sum(item['grade']=='中' for item in reviews)}，"
        f"差 {sum(item['grade']=='差' for item in reviews)}。"
    )
    report = f"""# Crum4 TLA corpus review

## Cohort and method

{summary}

- Top-10 scoring uses real `{model_name}` BGE vectors, ROUGE-L, and corpus lemma frequency. Coarse cap: `{limit}` per TLA.
- High-frequency cases are the ten highest corpus lemma/POS sentence counts. Low-frequency cases are selected from the bottom 30% of positive-count entries. POS groups include five each of NOUN, VERB, ADJ, and OTHER; all members of the requested homograph pairs are included.
- Evaluation labels are rule-based triage only: 好 if ROUGE-L ≥ 0.20, 中 if it is nonzero but lower, and 差 otherwise or if there are no candidates. This is not a human linguistic judgment; the explanation column makes the rule explicit.
- Local CDO comparison uses `dictionary/examples_sahidic.tab` and `examples_bohairic.tab`; “CDO 无数据” means neither local file contains a row for that TLA, not that the online service has no examples.

## TLA summary

| TLA | lemma | POS | gloss | 候选数 | top-1 译文 | 评价 | 理由 | CDO 对照 | 分层 |
|---|---|---|---|---:|---|---|---|---|---|
{chr(10).join(top_rows)}

## Local CDO example comparison

下表逐条列出本机 CDO TSV 中可用的 curated example，并与同一 TLA 的 Crum4 top-1 译文及语料引用并排比较。没有出现在表中的 TLA 在这两份本地 TSV 中无例句；这不代表在线 CDO 无数据。

| TLA | lemma | POS | CDO dialect | CDO Coptic sentence | CDO translation | CDO citation | Crum4 top-1 translation | Crum4 citation |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(cdo_comparison_rows) if cdo_comparison_rows else '| — | 本轮样本无本地 CDO 例句 | — | — | — | — | — | — | — |'}

## Top-10 results and CDO comparison

| TLA | rank | score | cosine | ROUGE-L | translation | corpus reference | CDO |
|---|---:|---:|---:|---:|---|---|---|
{chr(10).join(detail_rows)}
"""
    stats = {
        "tla_count": len(reviews),
        "nonzero_candidate_tlas": sum(bool(item["candidate_count"]) for item in reviews),
        "cdo_tlas": sum(bool(item["cdo_rows"]) for item in reviews),
        "grades": {grade: sum(item["grade"] == grade for item in reviews) for grade in ("好", "中", "差")},
        "reviews": reviews,
    }
    return report, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 1 <= args.limit <= 100:
        parser.error("--limit must be between 1 and 100.")
    report, stats = run_review(args.model, args.limit)
    args.output.write_text(report, encoding="utf-8")
    print(f"Reviewed TLAs: {stats['tla_count']}")
    print(f"Candidate matches: {stats['nonzero_candidate_tlas']}")
    print(f"Local CDO example entries: {stats['cdo_tlas']}")
    print(f"Grades: {stats['grades']}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
