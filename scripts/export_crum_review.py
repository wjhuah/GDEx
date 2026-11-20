pass
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
    with (config.derived / 'crum_bible_occurrences.jsonl').open(encoding='utf-8') as source, (config.derived / 'crum_review_queue.tsv').open('w', encoding='utf-8', newline='') as target:
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
            row = dict(id=record['id'], file=record['file'], line=record['line'], page=record['page'], reference=record['normalized_reference'], tla_candidates=','.join(record['tla_candidates']), reason=reason)
            writer.writerow(row)
            counts[reason] += 1
            if len(samples[reason]) < 3:
                samples[reason].append(row)
    md = ['', '', '', '', '', '|---|---:|']
    md.extend((f'| {reason} | {count} |' for reason, count in counts.items()))
    md.extend(['', '', '|---|---|---|---|---|'])
    for reason in counts:
        for row in samples[reason]:
            path = config.source / 'Crum_plain_text' / row['file']
            link = f"[{row['file']}:{row['line']}]({path}:{row['line']})"
            md.append(f"| {link} | {row['page']} | {row['reference']} | {row['tla_candidates']} | {reason} |")
    md.extend(['', f'{sum(counts.values()):,}'])
    (config.reports / 'CRUM_REVIEW_QUEUE.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    return dict(counts)
if __name__ == '__main__':
    print(export(Config()))
