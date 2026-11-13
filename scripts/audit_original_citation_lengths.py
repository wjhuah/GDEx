"""Separate literal quotation fragments, source qualifier scopes and full corpus spans."""

import sys, json, re, hashlib
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config
from coptic_examples.recommendation_policy import coptic_letters
from coptic_examples.text import COPTIC
from coptic_examples.qualifier_scope import rarity_scope

cfg = Config()
out = cfg.reports / "本地复盘/原数据八算法对照"
out.mkdir(parents=True, exist_ok=True)


def save(name, v):
    (out / name).write_text(json.dumps(v, ensure_ascii=False, indent=2) + "\n")


raw = list(map(json.loads, (cfg.derived / "complete_citations/citations.jsonl").open()))
fragments = []
for r in raw:
    paragraph = r["paragraph"]
    begin = paragraph.find(r["raw"])
    if begin < 0:
        continue
    tail = paragraph[begin + len(r["raw"]) :]
    nextref = re.search(r"⟦[^⟧]*⟧\{(?:bible|reference):", tail)
    if nextref:
        tail = tail[: nextref.start()]
    quotes = [
        q
        for q in re.findall(r"⟨([^⟩]*)⟩", tail)
        if COPTIC.search(q) and not q.startswith(("gloss:", "bullet:"))
    ]
    for i, q in enumerate(quotes):
        fragments.append(
            dict(
                citation_id=r["id"],
                file=r["file"],
                page=r["page"],
                reference=r["normalized_reference"],
                original_dialects=r["dialect_codes"],
                quote_index=i,
                literal_quote=q,
                letters=coptic_letters(q),
                bound_groups=len(q.split()),
                abbreviated=bool(re.search(r"[.…†⸗-]", q)),
                candidate_tlas=r["tla_candidates"],
                training_pairs=r["training_pairs"],
                scope="引文后、下一个出处前的显式原书片段；括注的其他方言也保留，方言范围另查",
            )
        )
(out / "原书短引语.jsonl").write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in fragments)
)
examples = []
entries = []
for dialect in ["sahidic", "bohairic"]:
    groups = defaultdict(list)
    for r in map(json.loads, (cfg.derived / f"model_rows_{dialect}.jsonl").open()):
        if r["y"] <= 0 or "crum" not in r["sources"]:
            continue
        c = r["candidate"]
        ranks = [p for p in r["ranks"] if p["source"] == "crum"]
        position = min((p.get("position", 1) for p in ranks), default=None)
        groups[r["tla"]].append(
            dict(
                dialect=dialect,
                tla=r["tla"],
                id=c["id"],
                letters=coptic_letters(c["sent"]),
                tt_tokens=len(c["tokens"]),
                bound_groups=len(c["sent"].split()),
                position=position,
            )
        )
    for tla, rr in groups.items():
        item = dict(dialect=dialect, tla=tla, n=len(rr))
        for unit in ["letters", "tt_tokens", "bound_groups"]:
            x = np.array([r[unit] for r in rr])
            mean = float(x.mean())
            sd = float(x.std())
            ordered = [r for r in rr if r["position"] is not None]
            rho = (
                float(
                    spearmanr(
                        [r["position"] for r in ordered], [r[unit] for r in ordered]
                    ).statistic
                )
                if len(ordered) >= 3 and len(set(r[unit] for r in ordered)) > 1
                else None
            )
            item[unit] = dict(
                mean=mean,
                sd=sd,
                median=float(np.median(x)),
                p90=float(np.quantile(x, 0.9)),
                maximum=int(max(x)),
                kept30=int(sum(x <= 30)),
                order_rho=rho,
            )
            for r in rr:
                r[unit + "_z"] = (r[unit] - mean) / sd if sd else 0
        examples += rr
        entries.append(item)
(out / "原数据逐句长度.jsonl").write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in examples)
)
save("原数据逐词条长度.json", entries)
rare = []
for path in sorted((cfg.source / "Crum_plain_text").glob("*.txt")):
    text = path.read_text()
    article = 0
    for paragraph_match in re.finditer(r"(?m)^¶[^\n]*", text):
        paragraph = paragraph_match[0]
        for m in re.finditer(r"\brare(?:ly)?\b", paragraph, re.I):
            before = paragraph[: m.start()]
            after = paragraph[m.end() :]
            head = before.rsplit("⟪", 1)
            scope = "待核定"
            codes = []
            form = ""
            # Only the complete immediate dialect tag cluster and parenthetical modifier count as explicit scope.
            mm = re.search(
                r"(?P<form>⟨[^⟩]+⟩)?\s*(?P<codes>(?:⟪[^⟫]+⟫\s*)+)\((?:very\s+|extremely\s+)?$",
                before,
                re.I,
            )
            if mm:
                scope = "紧随方言词形的少见标记"
                codes = re.findall(r"⟪([^⟫]+)⟫", mm["codes"])
                form = mm["form"] or ""
            elif re.search(r"\brare\s*$", before, re.I):
                scope = "待核定"
            article_start = text.rfind("___", 0, paragraph_match.start())
            article_key = path.name + ":" + str(article_start)
            article_text = text[
                article_start + 3 if article_start >= 0 else 0 : paragraph_match.start()
                + len(paragraph)
            ]
            headword_match = re.search(r"(?m)^¶\s*⟨([^⟩]+)⟩", article_text)
            headword = headword_match[1] if headword_match else ""
            local_scope = rarity_scope(before, after)
            rare.append(
                dict(
                    local_scope=local_scope,
                    article_key=article_key,
                    headword=headword,
                    file=path.name,
                    file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    offset=paragraph_match.start() + m.start(),
                    qualifier=m[0],
                    scope=scope,
                    dialect_codes=codes,
                    form=form,
                    source_paragraph=paragraph,
                )
            )
(out / "少见标记逐处范围.jsonl").write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rare)
)
summary = dict(
    raw_citations=len(raw),
    printed_fragment_count=len(fragments),
    printed_fragments_without_abbreviation=sum(not r["abbreviated"] for r in fragments),
    printed_mean_letters=(
        float(np.mean([r["letters"] for r in fragments])) if fragments else None
    ),
    rare_mentions=len(rare),
    rare_scope_counts=dict(Counter(r["scope"] for r in rare)),
    explicit_S_rare_mentions=sum("S" in r["dialect_codes"] for r in rare),
)
summary["local_scope_counts"] = dict(Counter(r["local_scope"]["scope"] for r in rare))
summary["local_scope_S_form_mentions"] = sum(
    "S" in r["local_scope"]["dialect_codes"] and r["local_scope"]["scope"] == "方言词形"
    for r in rare
)
summary["local_scope_S_article_blocks"] = len(
    set(
        r["article_key"]
        for r in rare
        if "S" in r["local_scope"]["dialect_codes"]
        and r["local_scope"]["scope"] == "方言词形"
    )
)
summary["explicit_S_rare_article_blocks"] = len(
    set(r["article_key"] for r in rare if "S" in r["dialect_codes"])
)
summary["corpus"] = {}
for d in ["sahidic", "bohairic"]:
    rr = [r for r in examples if r["dialect"] == d]
    summary["corpus"][d] = dict(pairs=len(rr), entries=len(set(r["tla"] for r in rr)))
    for unit in ["letters", "tt_tokens", "bound_groups"]:
        rho = [
            r[unit]["order_rho"]
            for r in entries
            if r["dialect"] == d and r[unit]["order_rho"] is not None
        ]
        summary["corpus"][d][unit] = dict(
            mean=float(np.mean([r[unit] for r in rr])),
            median=float(np.median([r[unit] for r in rr])),
            kept30=sum(r[unit] <= 30 for r in rr),
            order_rho_median=float(np.median(rho)) if rho else None,
        )
save("引文句长与方言摘要.json", summary)
print(json.dumps(summary, ensure_ascii=False, indent=2))
