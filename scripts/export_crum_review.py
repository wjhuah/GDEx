#!/usr/bin/env python3
"""Export unresolved interpretations without discarding any raw citation."""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from coptic_examples.config import Config


def export(config):
    counts = Counter()
    samples = defaultdict(list)
    fields = ['id', 'file', 'line', 'page', 'reference', 'tla_candidates', 'reason']
    with (config.derived / 'crum_bible_occurrences.jsonl').open(encoding='utf-8') as source, \
            (config.derived / 'crum_review_queue.tsv').open('w', encoding='utf-8', newline='') as target:
        writer = csv.DictWriter(target, fields, delimiter='\t')
        writer.writeheader()
        for line in source:
            record = json.loads(line)
            reasons = []
            if not record['tla_candidates']:
                reasons.append('TLA unresolved')
            elif len(record['tla_candidates']) > 1:
                reasons.append('TLA ambiguous')
            if record['link_status'] != 'linked':
                reasons.append(record['link_status'])
            if not reasons:
                continue
            reason = '; '.join(reasons)
            row = dict(id=record['id'], file=record['file'], line=record['line'], page=record['page'],
                       reference=record['normalized_reference'],
                       tla_candidates=','.join(record['tla_candidates']), reason=reason)
            writer.writerow(row)
            counts[reason] += 1
            if len(samples[reason]) < 3:
                samples[reason].append(row)
    md = ['# Crum 引用待核对队列', '',
          '完整原始标记已提取。以下是解释/对齐尚未确认的代表项，不能把它们伪装成正确训练句。整章或书卷级引用本身不一定是错误。完整队列见 data/derived/crum_review_queue.tsv。', '',
          '| 类别 | 数量 |', '|---|---:|']
    md.extend(f'| {reason} | {count} |' for reason, count in counts.items())
    md.extend(['', '| 原文件位置 | 印刷页 | 引用 | TLA 候选 | 状态 |', '|---|---|---|---|---|'])
    for reason in counts:
        for row in samples[reason]:
            path = config.source / 'Crum_plain_text' / row['file']
            link = f"[{row['file']}:{row['line']}]({path}:{row['line']})"
            md.append(f"| {link} | {row['page']} | {row['reference']} | {row['tla_candidates']} | {reason} |")
    md.extend(['', f'合计 {sum(counts.values()):,} 条。重建命令：`python scripts/export_crum_review.py`。'])
    (config.reports / 'CRUM_REVIEW_QUEUE.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    return dict(counts)


if __name__ == '__main__':
    print(export(Config()))
