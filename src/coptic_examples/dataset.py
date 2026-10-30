"""Rebuild supervision from original manual references and audited Crum references."""
import csv
import hashlib
import json
import sqlite3
from io import StringIO
from collections import Counter,defaultdict
from pathlib import Path
from .annotations import load_manual,resolve_manual,forbidden_rules,forbidden,contains_lemma
from .config import write_json,database
from .corpus import Corpus
from .lexicon import load
from .text import lemma_key,norm,variants,real_translation
from .dialects import Dialects


def text_key(row):
    return hashlib.sha256(norm(row['sent']).replace(' ','').casefold().encode()).hexdigest()


def read_rows(config):
    with (config.derived/'supervision.jsonl').open(encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def build(config):
    corpus=Corpus(config);lex=load(config.lexicon)
    registry=Dialects()
    annotations=load_manual(config.manual)
    resolved,audit=resolve_manual(annotations,corpus)
    rules=forbidden_rules(annotations)
    stats=Counter(manual_rows=len(annotations),manual_matched_rows=sum(a['status']=='matched' for a in audit),
                  manual_unresolved_rows=sum(a['status']=='unresolved' for a in audit),manual_rule_rows=sum(a['status']=='rule_only' for a in audit))
    config.derived.mkdir(parents=True,exist_ok=True)
    write_json(config.reports/'manual_alignment.json',audit)
    examples=[]
    positive_ids=defaultdict(set)
    manual_by_key={}
    for r in resolved:
        tla,a,c=r['TLA'],r['annotation'],r['candidate']
        if tla not in lex:
            stats['manual_unknown_tla']+=1;continue
        key=(tla,text_key(c))
        priority=int(a['priority']) if a['priority'] in {'1','2','3'} else None
        # Missing priority is positive evidence of inclusion, not an invented rank 3.
        if priority is None:
            stats['manual_unranked_resolved']+=1
        grade={1:30,2:20,3:10}.get(priority)
        record=dict(TLA=tla,lemma=lex[tla].lemma,gloss=lex[tla].gloss,group=f'{tla}|manual',
                    evidence='manual',manual_row=a['row_id'],priority=priority,crum_rank=None,
                    grade=grade,signed_label=grade/10 if grade else None,
                    candidate=c,text_key=text_key(c),weight=1.0)
        positive_ids[tla].add(c['id'])
        if key not in manual_by_key or (grade or 0)>(manual_by_key[key]['grade'] or 0):
            manual_by_key[key]=record
    examples.extend(manual_by_key.values())
    # Keep negatives from explicit forbidden rules; never infer negatives from absence.
    seen=set(manual_by_key)
    all_forbidden=[]
    for tla,rs in sorted(rules.items()):
        if tla not in lex:
            continue
        pool=[]
        for c in corpus.candidates(lex[tla].lemma,lex[tla].pos or None):
            key=(tla,text_key(c))
            if key in seen:
                if forbidden(c,rs,True): stats['manual_rule_positive_conflicts']+=1
                continue
            if forbidden(c,rs,c['id'] in positive_ids[tla]):
                stats['explicit_rule_negative_spans']+=1
                all_forbidden.append(dict(TLA=tla,sentence_id=c['id'],signed_label=-1,evidence='manual_forbidden_rule'))
                if real_translation(c['translation']):pool.append(c)
        # Bound training imbalance, never corpus coverage. Full rule hits are separately exported.
        pool.sort(key=lambda c:hashlib.sha256(c['id'].encode()).hexdigest())
        for c in pool[:20]:
                examples.append(dict(TLA=tla,lemma=lex[tla].lemma,gloss=lex[tla].gloss,group=f'{tla}|manual',
                                     evidence='manual_forbidden_rule',grade=0,signed_label=-1,weight=1.0,
                                     candidate=c,text_key=text_key(c),crum_rank=None,manual_row=None,priority=None))
                seen.add((tla,text_key(c)));stats['explicit_rule_negatives_selected_for_training']+=1
    with (config.derived/'forbidden_matches.tsv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['TLA','sentence_id','signed_label','evidence'],delimiter='\t');w.writeheader();w.writerows(all_forbidden)
    # Revalidate the existing OCR->Bible link table against actual TT source and target lemma.
    # It is a legacy derived input, not a newly asserted gold annotation.
    rebuilt=config.derived/'crum_senses_matched.tsv'
    crum_path=rebuilt if rebuilt.exists() else config.source/'crum3/crum_senses_matched_v15.tsv'
    write_json(config.reports/'crum_link_input.json',dict(path=str(crum_path),sha256=hashlib.sha256(crum_path.read_bytes()).hexdigest() if crum_path.exists() else None,
                                                       source='optional_legacy_comparison_only; current style training uses complete crum_bible.sqlite3'))
    crum_audit=[]
    with (crum_path.open(encoding='utf-8') if crum_path.exists() else StringIO('')) as f, corpus.connect() as db:
        for number,r in enumerate(csv.DictReader(f,delimiter='\t'),2):
            tla=r['TLA'];stats['crum_link_rows']+=1
            if tla not in lex:
                stats['crum_unknown_tla']+=1;continue
            try:
                source=Path(r['corpus_file']).relative_to(config.source/'corpora').as_posix()
            except ValueError:
                # A relocated archive may retain an old machine's absolute prefix.
                source=r['corpus_file'].split('/corpora/',1)[-1]
            matches=[json.loads(x[0]) for x in db.execute('SELECT payload FROM sentence WHERE source=? AND CAST(chapter AS INT)=? AND CAST(verse AS INT)=?',(source,int(r['chapter']),int(r['verse'])))]
            # Dictionary spelling/inflection aliases count, but raw substrings do not.
            valid=[c for c in matches if any(contains_lemma(c,form) for form in lex[tla].forms)]
            if not valid:
                status='missing_target_lemma' if matches else 'unresolved_reference'
                stats['crum_'+status]+=1
                crum_audit.append(dict(row=number,TLA=tla,source=source,chapter=r['chapter'],verse=r['verse'],status=status))
                continue
            for c in valid:
                if not real_translation(c['translation']):
                    stats['crum_no_translation']+=1
                    continue
                if forbidden(c,rules[tla],c['id'] in positive_ids[tla]):
                    stats['crum_forbidden_rejected']+=1;continue
                rank=int(r['rank'])
                examples.append(dict(TLA=tla,lemma=lex[tla].lemma,gloss=lex[tla].gloss,
                                     group=f"{tla}|crum:{r['sense_idx']}",evidence='crum_citation_order_weak',
                                     grade=max(10,31-rank),signed_label=None,crum_rank=rank,
                                     crum_link_row=number,manual_row=None,priority=None,
                                     weight=0.35,candidate=c,text_key=text_key(c)))
    # Dedupe within query, record conflicting labels and prefer strongest grade deterministically.
    unique={};conflicts=[]
    for r in examples:
        key=(r['group'],r['text_key'])
        if key in unique:
            old=unique[key]
            if old['grade']!=r['grade']:
                conflicts.append(dict(group=r['group'],text_key=r['text_key'],grades=[old['grade'],r['grade']]))
            if (r['grade'] or 0)>(old['grade'] or 0):unique[key]=r
        else:unique[key]=r
    rows=sorted(unique.values(),key=lambda r:(r['group'],r['candidate']['id']))
    for r in rows:
        r['dialect_id']=r['candidate'].get('dialect_id','unknown')
        r['group']+='|'+r['dialect_id']
    with (config.derived/'supervision.jsonl').open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    write_json(config.reports/'crum_alignment_rejections.json',crum_audit)
    write_json(config.reports/'label_conflicts.json',conflicts)
    # Map dictionary entries to all corpus spans, including native form variants, without ranking caps.
    # This is a retrieval index, not a labeled dataset.
    mwe_rows=[];mwe_first=defaultdict(list)
    for e in lex.values():
        if ' ' in e.lemma:
            components=[set(variants(x)) for x in e.lemma.split()]
            for first in components[0]:mwe_first[first].append((e,components))
    # Scan once, rather than repeatedly decoding a whole corpus for high-frequency MWE heads.
    for c in corpus.all():
        ts=c['tokens']
        for i,t in enumerate(ts):
            for e,parts in mwe_first.get(t['lemma'],[]):
                span=ts[i:i+len(parts)]
                if len(span)==len(parts) and (not e.pos or span[0]['pos']==e.pos) and all(tok['lemma'] in forms for tok,forms in zip(span,parts)):
                    mwe_rows.append((e.tla,c['id'],'mwe_contiguous'))
    with database(config.index) as db:
        db.executescript('DROP TABLE IF EXISTS entry_form; DROP TABLE IF EXISTS candidate; CREATE TABLE entry_form(tla TEXT,lemma TEXT,pos TEXT,match_kind TEXT,dialect TEXT); CREATE INDEX form_lookup ON entry_form(lemma,pos,dialect); CREATE TABLE candidate(tla TEXT,sentence_id TEXT,dialect TEXT,match_kind TEXT,PRIMARY KEY(tla,sentence_id)); CREATE INDEX candidates_by_sentence ON candidate(sentence_id); CREATE INDEX candidates_by_dialect ON candidate(tla,dialect);')
        form_rows=[]
        for e in lex.values():
            if not e.pos:continue
            for form in e.forms:
                if ' ' in form:continue
                codes=(e.form_dialects or {}).get(form,[])
                dialects=registry.from_codes(codes) if codes else list(registry.data)
                for variant in variants(form):
                    for dialect in dialects:form_rows.append((e.tla,variant,e.pos,'exact_form' if variant==form else 'vowel_variant',dialect))
        db.executemany('INSERT INTO entry_form VALUES(?,?,?,?,?)',sorted(set(form_rows)))
        db.execute("INSERT OR IGNORE INTO candidate SELECT f.tla,t.sentence_id,s.dialect,MIN(f.match_kind) FROM entry_form f JOIN token t ON f.lemma=t.lemma AND f.pos=t.pos JOIN sentence s ON s.id=t.sentence_id AND s.dialect=f.dialect GROUP BY f.tla,t.sentence_id")
        # MWEs require contiguous lemma sequences; they cannot be found by one token equality.
        for tla,sid,kind in mwe_rows:
            dialect=db.execute('SELECT dialect FROM sentence WHERE id=?',(sid,)).fetchone()[0]
            codes=(lex[tla].form_dialects or {}).get(lex[tla].lemma,[])
            if (not codes and dialect in registry.data) or dialect in registry.from_codes(codes):db.execute('INSERT OR IGNORE INTO candidate VALUES(?,?,?,?)',(tla,sid,dialect,kind))
        # Explicit manual string-match cases can target a different corpus lemma/POS.
        db.executemany('INSERT OR IGNORE INTO candidate VALUES(?,?,?,?)',[(r['TLA'],r['candidate']['id'],r['candidate']['dialect_id'],'manual_reference_override') for r in resolved if r['TLA'] in lex and r['candidate']['dialect_id'] in registry.data])
        stats['dictionary_entries']=len(lex)
        stats['dictionary_entries_with_candidates']=db.execute('SELECT COUNT(DISTINCT tla) FROM candidate').fetchone()[0]
        stats['dictionary_candidate_links']=db.execute('SELECT COUNT(*) FROM candidate').fetchone()[0]
        for dialect,n in db.execute('SELECT dialect,COUNT(*) FROM candidate GROUP BY dialect'):stats['candidate_links_'+dialect]=n
    stats.update(supervision_rows=len(rows),manual_ranked=sum(r['evidence']=='manual' and r['grade'] is not None for r in rows),
                 crum_validated_weak_rows=sum(r['evidence'].startswith('crum') for r in rows),label_conflicts=len(conflicts))
    write_json(config.reports/'dataset_inventory.json',dict(stats))
    # Every original span, all fields including licenses and unknown translations, is exported.
    with (config.derived/'all_examples.jsonl').open('w',encoding='utf-8') as f:
        for c in corpus.all():f.write(json.dumps(c,ensure_ascii=False)+'\n')
    return dict(stats)
