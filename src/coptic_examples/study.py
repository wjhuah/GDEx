"""Pre-registered C992/C998 contrast, grounded only in expert-selected references."""

import csv
import hashlib
import json
import numpy as np
from .annotations import load_manual, resolve_manual
from .config import write_json
from .embeddings import Embeddings, MODEL
from .lexicon import load
from .text import real_translation, sm_freq


def run(config):
    from .corpus import Corpus

    corpus = Corpus(config)
    annotations = load_manual(config.manual)
    target = [
        a
        for a in annotations
        if a["TLA"] in {"C992", "C998"} and a["priority"] in {"1", "2", "3"}
    ]
    resolved, audit = resolve_manual(target, corpus)
    if any(a["status"] != "matched" for a in audit):
        write_json(config.reports / "hypothesis_unresolved.json", audit)
        raise RuntimeError(
            "Some expert anchors did not match; inspect hypothesis_unresolved.json"
        )
    # Multiple corpus witnesses do not inflate experimental N.
    unique = {}
    for r in resolved:
        a, c = r["annotation"], r["candidate"]
        key = (r["TLA"], a["row_id"], c["sent"], c["translation"])
        unique.setdefault(key, r)
    anchors = list(unique.values())
    if not anchors or not all(
        real_translation(r["candidate"]["translation"]) for r in anchors
    ):
        raise RuntimeError("A resolved expert anchor has no English translation")
    lex = load(config.lexicon)
    embed = Embeddings(config.vectors)
    labels = ["C992", "C998"]
    definitions = [lex[t].gloss for t in labels]
    q = embed.encode(definitions, "definition")
    v = embed.encode([r["candidate"]["translation"] for r in anchors])
    scores = v @ q.T
    rows = []
    for r, cs in zip(anchors, scores, strict=True):
        a, c = r["annotation"], r["candidate"]
        own = r["TLA"]
        other = next(t for t in labels if t != own)
        own_score = float(cs[labels.index(own)])
        other_score = float(cs[labels.index(other)])
        rows.append(
            dict(
                expert_tla=own,
                cross_tla=other,
                manual_row=a["row_id"],
                priority=a["priority"],
                signed_cross_label=-1,
                cross_label_status="inferred_wrong_sense_NOT_expert_cross_annotation",
                sentence_id=c["id"],
                source=c["source"],
                urn=c["urn"],
                chapter=c["chapter"],
                verse=c["verse"],
                sent=c["sent"],
                translation=c["translation"],
                own_cosine=own_score,
                cross_cosine=other_score,
                cosine_margin=own_score - other_score,
                own_sm_freq=sm_freq(lex[own].gloss, c["translation"]),
                cross_sm_freq=sm_freq(lex[other].gloss, c["translation"]),
            )
        )
    result = dict(
        protocol=dict(
            model=MODEL,
            definitions=dict(zip(labels, definitions)),
            anchors="Only references selected in citations_manual_merged.tab; current TT text",
            comparison="cosine(own gloss,translation) > cosine(other gloss,translation)",
            no_training=True,
            no_threshold_fitting=True,
            cross_labels="inferred -1 for this disjoint pair; not human-confirmed new negative labels",
            unit="unique expert reference + Coptic span + translation; duplicated witnesses collapsed",
            caveat="Translation similarity supports sense contrast; cannot align a specific Coptic token when both senses occur",
        ),
        manual_sha256=hashlib.sha256(config.manual.read_bytes()).hexdigest(),
        anchor_count=len(rows),
        correct_contrasts=sum(r["cosine_margin"] > 0 for r in rows),
        mean_margin=float(np.mean([r["cosine_margin"] for r in rows])),
        rows=rows,
        audit=audit,
    )
    write_json(config.reports / "barley_father.json", result)
    config.derived.mkdir(parents=True, exist_ok=True)
    with (config.derived / "barley_father_cross_negatives.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    md = [
        "# Amir 假设：barley / father 交叉负例验证",
        "",
        "本实验先回到原始人工表，按作品 URN、章、节和目标 lemma 找回当前语料中的句子，再计算真实 BGE 相似度。没有训练，也没有为这四个例子调阈值。",
        "",
        f"模型：`{MODEL}`。定义：C992 = `{lex['C992'].gloss}`；C998 = `{lex['C998'].gloss}`。",
        "",
        "| 人工正例 | 引用 | 译文 | 自身义项 cosine | 另一义项 cosine | 差值 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for r in rows:
        ref = f"{r['urn']} {r['chapter']}:{r['verse']}"
        md.append(
            f"| {r['expert_tla']} | {ref} | {r['translation'].replace('|','/')} | {r['own_cosine']:.4f} | {r['cross_cosine']:.4f} | {r['cosine_margin']:+.4f} |"
        )
    md += [
        "",
        f"有效人工锚点 **{len(rows)}**，自身义项相似度较高的有 **{result['correct_contrasts']}**，平均差值 **{result['mean_margin']:.4f}**。",
        "",
        "## 结论与边界",
        "",
        (
            "这组已核对引用支持 Amir 在 C992/C998 上的思路：同形词的好例句，可以作为互不相容义项的候选负例。"
            if result["correct_contrasts"] == len(rows)
            else "该组对照没有全部通过，不能把 Amir 的建议当作已证实规则；逐条检查上表。"
        ),
        "原人工表没有把这些句子交叉标为另一 TLA 的 -1。导出的 -1 是本研究的派生假设标签，单独标明来源；不能伪称专家新标注。",
        "这只能验证四个锚点的义项对照；不能证明加入负例会提升排序 NDCG，更不能推广为“所有同形 TLA 互为负例”。father 与 father-as-title 的近义关系必须保留不确定性。",
        "句子可能同时出现父亲和大麦；整句英文向量无法决定具体哪个 Coptic token 对应哪个义项。因此自动扩展需要目标词出现检查、定义关系检查、去重、冲突检查和独立评测。",
        "排序训练中的负例采用非负 relevance=0；界面/证据表可保留 signed_label=-1。无人工标注的候选仍是 unknown，不自动记为负例。",
        "",
        "这里“负例”指对目标义项不合适，不要求余弦相似度低于 0；本次交叉 cosine 仍为正值。",
        "具体 Coptic 原文、匹配路径、人工表行号、原始数值见 `barley_father.json`，重跑命令：`python run.py study`。",
    ]
    config.reports.mkdir(parents=True, exist_ok=True)
    (config.reports / "BARLEY_FATHER.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    return {k: v for k, v in result.items() if k not in {"rows", "audit"}}
