pass
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
    target = [a for a in annotations if a['TLA'] in {'C992', 'C998'} and a['priority'] in {'1', '2', '3'}]
    resolved, audit = resolve_manual(target, corpus)
    if any((a['status'] != 'matched' for a in audit)):
        write_json(config.reports / 'hypothesis_unresolved.json', audit)
        raise RuntimeError('Some expert anchors did not match; inspect hypothesis_unresolved.json')
    unique = {}
    for r in resolved:
        a, c = (r['annotation'], r['candidate'])
        key = (r['TLA'], a['row_id'], c['sent'], c['translation'])
        unique.setdefault(key, r)
    anchors = list(unique.values())
    if not anchors or not all((real_translation(r['candidate']['translation']) for r in anchors)):
        raise RuntimeError('A resolved expert anchor has no English translation')
    lex = load(config.lexicon)
    embed = Embeddings(config.vectors)
    labels = ['C992', 'C998']
    definitions = [lex[t].gloss for t in labels]
    q = embed.encode(definitions, 'definition')
    v = embed.encode([r['candidate']['translation'] for r in anchors])
    scores = v @ q.T
    rows = []
    for r, cs in zip(anchors, scores, strict=True):
        a, c = (r['annotation'], r['candidate'])
        own = r['TLA']
        other = next((t for t in labels if t != own))
        own_score = float(cs[labels.index(own)])
        other_score = float(cs[labels.index(other)])
        rows.append(dict(expert_tla=own, cross_tla=other, manual_row=a['row_id'], priority=a['priority'], signed_cross_label=-1, cross_label_status='inferred_wrong_sense_NOT_expert_cross_annotation', sentence_id=c['id'], source=c['source'], urn=c['urn'], chapter=c['chapter'], verse=c['verse'], sent=c['sent'], translation=c['translation'], own_cosine=own_score, cross_cosine=other_score, cosine_margin=own_score - other_score, own_sm_freq=sm_freq(lex[own].gloss, c['translation']), cross_sm_freq=sm_freq(lex[other].gloss, c['translation'])))
    result = dict(protocol=dict(model=MODEL, definitions=dict(zip(labels, definitions)), anchors='Only references selected in citations_manual_merged.tab; current TT text', comparison='cosine(own gloss,translation) > cosine(other gloss,translation)', no_training=True, no_threshold_fitting=True, cross_labels='inferred -1 for this disjoint pair; not human-confirmed new negative labels', unit='unique expert reference + Coptic span + translation; duplicated witnesses collapsed', caveat='Translation similarity supports sense contrast; cannot align a specific Coptic token when both senses occur'), manual_sha256=hashlib.sha256(config.manual.read_bytes()).hexdigest(), anchor_count=len(rows), correct_contrasts=sum((r['cosine_margin'] > 0 for r in rows)), mean_margin=float(np.mean([r['cosine_margin'] for r in rows])), rows=rows, audit=audit)
    write_json(config.reports / 'barley_father.json', result)
    config.derived.mkdir(parents=True, exist_ok=True)
    with (config.derived / 'barley_father_cross_negatives.tsv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    md = ['local_f929084d/local_8f7951d2', '', '', '', f"Model: `{MODEL}`. Definition: C992 = `{lex['C992'].gloss}`；C998 = `{lex['C998'].gloss}`。", '', '', '|---|---|---|---:|---:|---:|']
    for r in rows:
        ref = f"{r['urn']} {r['chapter']}:{r['verse']}"
        md.append(f"| {r['expert_tla']} | {ref} | {r['translation'].replace('|', '/')} | {r['own_cosine']:.4f} | {r['cross_cosine']:.4f} | {r['cosine_margin']:+.4f} |")
    md += ['', f"Labeled anchors **{len(rows)}**; higher own-sense similarity **{result['correct_contrasts']}**; mean difference **{result['mean_margin']:.4f}**。", '', '', '', '' if result['correct_contrasts'] == len(rows) else '', '', '', '', '', '', '', '']
    config.reports.mkdir(parents=True, exist_ok=True)
    (config.reports / 'BARLEY_FATHER.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    return {k: v for k, v in result.items() if k not in {'rows', 'audit'}}
