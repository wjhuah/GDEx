"""Target-centered information and interpretable syntactic coverage."""
import itertools
import math
from collections import Counter
import numpy as np
from .text import lemma_key,norm,variants

CONTENT={'NOUN','ADJ','VERB','ADV','NUM'}


def target_indices(candidate,lemma,forms=()):
    keys={v for form in [lemma,*forms] for v in variants(form)}
    if not keys:keys={norm(lemma).casefold()}  # language-neutral unit fixtures
    return [i for i,t in enumerate(candidate['tokens']) if (lemma_key(t['lemma']) or norm(t['lemma']).casefold()) in keys]


def profile(candidate,lemma,forms=()):
    ts=candidate['tokens'];byid={t['id']:t for t in ts};children={t['id']:[] for t in ts}
    for t in ts:
        if t['head'] in children:children[t['head']].append(t)
    targets=target_indices(candidate,lemma,forms);signature=set();descriptors=0;roles=set();clause_nodes=set()
    for index in targets:
        t=ts[index];signature.add('role:'+ (t['dep'] or 'unknown'))
        parent=byid.get(t['head']);signature.add('head_pos:'+(parent['pos'] if parent else 'ROOT_OR_EXTERNAL'))
        pred=t if t['pos']=='VERB' else parent
        for child in children[t['id']]:
            if child['dep'] in {'amod','nmod','acl','acl:relcl','appos'}:
                descriptors+=1;signature.add('modifier:'+child['dep'])
        if parent and t['dep'].startswith('nsubj') and parent['pos'] in {'ADJ','NOUN'}:
            descriptors+=1;signature.add('construction:property_predication')
        node=t;seen=set();subordinate=False
        while node['id'] not in seen:
            seen.add(node['id']);clause_nodes.add(node['id'])
            if node['dep'] in {'acl','acl:relcl','advcl','ccomp','xcomp','csubj'}:subordinate=True
            if node['head'] not in byid:break
            node=byid[node['head']]
        signature.add('clause:'+('subordinate' if subordinate else 'matrix_or_fragment'))
        if pred:
            args={c['dep'].split(':')[0] for c in children[pred['id']] if c['dep'].split(':')[0] in {'nsubj','obj','iobj','obl','ccomp','xcomp'}}
            roles.update(args)
            signature.add('frame:'+pred['pos']+'+'+'+'.join(sorted(args)))
            for c in children[pred['id']]:
                if c['dep']=='aux':signature.add('aux:'+(c.get('xpos') or c['pos']))
                if c.get('xpos')=='NEG' or c.get('xpos','').startswith('ANEG'):signature.add('polarity:negative')
    context=[t for i,t in enumerate(ts) if i not in targets and t['pos'] in CONTENT]
    proper=sum(t['pos']=='PROPN' for t in ts)
    information=.45*min(1,descriptors/3)+.35*min(1,len(context)/6)+.20*min(1,len(roles)/3)
    features=dict(context_content_count=min(12,len(context)),target_descriptor_count=min(6,descriptors),
                  filled_argument_roles=len(roles),target_occurrences=len(targets),
                  context_content_diversity=len({t['lemma'] for t in context})/max(1,len(context)),
                  proper_name_ratio=proper/max(1,len(ts)),information_score=information,
                  dependency_target_found=int(bool(targets)))
    return dict(signature=sorted(signature),features=features,target_indices=targets,
                annotation_status=candidate.get('parsing','unspecified'))


def signature_weight(feature):
    return {'role':3.,'frame':2.,'construction':2.,'clause':1.,'head_pos':1.,'modifier':.75,'aux':.75,'polarity':.5}.get(feature.split(':')[0],.5)


def lexical_set(row):
    return {t['lemma'] for t in row['tokens'] if t['pos'] in CONTENT and t['lemma']}


def similarity(a,b):
    if norm(a['sent']).replace(' ','')==norm(b['sent']).replace(' ',''):return 1.
    x,y=lexical_set(a),lexical_set(b)
    return len(x&y)/len(x|y) if x|y else 0.


def metrics(selected,universe):
    allfeatures=set().union(*(set(r['syntax']['signature']) for r in universe)) if universe else set()
    covered=set().union(*(set(r['syntax']['signature']) for r in selected)) if selected else set()
    denominator=sum(signature_weight(x) for x in allfeatures)
    redundancy=[similarity(a,b) for a,b in itertools.combinations(selected,2)]
    return dict(mean_quality=float(np.mean([r.get('acceptance_score',r['shortlist_score']) for r in selected])) if selected else 0.,
                mean_combined_score=float(np.mean([r['shortlist_score'] for r in selected])) if selected else 0.,
                weighted_syntax_coverage=sum(signature_weight(x) for x in covered)/denominator if denominator else 0.,
                signature_features=len(covered),mean_lexical_redundancy=float(np.mean(redundancy)) if redundancy else 0.,
                target_roles=sorted(x for x in covered if x.startswith('role:')),
                observed_signature=sorted(covered))


def utility(selected,universe):
    m=metrics(selected,universe)
    return .60*m['mean_quality']+.30*m['weighted_syntax_coverage']-.10*m['mean_lexical_redundancy']


def choose(rows,k,method='coverage'):
    if len({r['dialect_id'] for r in rows})>1:raise ValueError('Example sets cannot mix dialects')
    if len({r.get('tla') for r in rows})>1:raise ValueError('Example sets cannot mix TLA queries')
    if not rows:return []
    k=min(k,len(rows));ranked=sorted(rows,key=lambda r:(-r['shortlist_score'],r['id']))
    if method=='rank':return ranked[:k]
    if method=='coverage' and k<=3 and len(rows)<=20:
        return list(max(itertools.combinations(ranked,k),key=lambda group:utility(group,rows)))
    if method=='kmeans':
        from sklearn.cluster import KMeans
        columns=sorted({s for r in rows for s in r['syntax']['signature']})
        if not columns:return ranked[:k]
        x=np.asarray([[math.sqrt(signature_weight(s)) if s in r['syntax']['signature'] else 0. for s in columns] for r in ranked])
        clusters=min(k,len(np.unique(x,axis=0)))
        labels=KMeans(n_clusters=clusters,n_init=10,random_state=42).fit_predict(x)
        chosen=[next(r for r,l in zip(ranked,labels) if l==cluster) for cluster in range(clusters)]
        chosen_ids={r['id'] for r in chosen}
        return sorted(chosen+[r for r in ranked if r['id'] not in chosen_ids][:k-len(chosen)],key=lambda r:-r['shortlist_score'])
    # Quality-bounded marginal coverage selection for the ten-candidate stage.
    chosen=[]
    for _ in range(k):
        available=[r for r in ranked if r not in chosen]
        best=max(available,key=lambda r:(utility(chosen+[r],rows),r['shortlist_score'],r['id']))
        chosen.append(best)
    return sorted(chosen,key=lambda r:-r['shortlist_score'])
