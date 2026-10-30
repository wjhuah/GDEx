"""Fixed surface frequency statistics and native dependency features for both fit and search."""
import math
import re
from collections import Counter
from .text import norm,variants,sm_freq,real_translation

STOP=set('ⲙ ⲛ ⲉ ϩⲛ ϩⲓ ⲛⲧⲉ ⲛⲧⲁ ⲡ ⲧ ⲛⲓ ⲛⲉ ⲟⲩ ⲡⲉ ⲧⲉ ⲁ ⲉⲣ ⲣ ϣⲁ ⲉⲃⲟⲗ'.split())
GDEX='sent_len n_chars avg_word_length max_word_length min_word_length n_commas n_punctuation n_parentheses n_digits is_complete_sentence target_position target_position_ratio target_repetition target_at_edge mean_word_frequency min_word_frequency mean_log_word_frequency rare_word_ratio unique_word_ratio top1k_ratio top5k_ratio top10k_ratio trans_len has_translation'.split()
UD='has_dependency n_tokens n_subjects n_objects n_clauses n_subordinate_clauses n_coordination tree_depth mean_dep_distance has_verb pronoun_ratio target_in_main_clause'.split()
SM=['sm_freq','sm_cosine']


def mean(values):return sum(values)/len(values) if values else 0.0


class Features:
    def __init__(self,frequencies):
        self.freq=frequencies
        ranked=sorted(frequencies,key=lambda w:(-frequencies[w],w))
        self.top={n:set(ranked[:n]) for n in [1000,5000,10000]}

    def extract(self,candidate,lemma):
        st=norm(candidate['sent']);words=st.split();n=len(words);lengths=list(map(len,words))
        positions=[i for i,w in enumerate(words,1) if any(v in w for v in variants(lemma))]
        lexical=[w for w in words if w not in STOP];freqs=[self.freq.get(w,0) for w in lexical]
        positive=[x for x in freqs if x>0]
        tr=norm(candidate['translation']).split()
        f=dict(sent_len=n,n_chars=len(st),avg_word_length=mean(lengths),max_word_length=max(lengths,default=0),min_word_length=min(lengths,default=0),
               n_commas=st.count(',')+st.count('⸲'),n_punctuation=sum(c in '.,;:!?·…⸲⳹⳸' for c in st),n_parentheses=sum(c in '()[]{}' for c in st),n_digits=sum(c.isdigit() for c in st),
               is_complete_sentence=int(bool(re.search(r'[.!?;·…]$',st))),target_position=positions[0] if positions else 0,
               target_position_ratio=positions[0]/n if positions and n else 0,target_repetition=len(positions),target_at_edge=int(bool(positions and positions[0] in {1,n})),
               mean_word_frequency=mean(positive),min_word_frequency=min(positive,default=0),mean_log_word_frequency=mean([math.log(x) for x in positive]),rare_word_ratio=mean([x<=5 for x in positive]),unique_word_ratio=len(set(words))/n if n else 0,
               trans_len=len(tr),has_translation=int(real_translation(candidate['translation'])))
        for limit in [1000,5000,10000]:f[f'top{limit//1000}k_ratio']=sum(w in self.top[limit] for w in lexical)/max(1,len(lexical))
        tokens=candidate['tokens'];counts=Counter(t['dep'] for t in tokens);byid={t['id']:t for t in tokens};order={t['id']:i for i,t in enumerate(tokens)}
        depths=[]
        for t in tokens:
            seen=set();node=t
            while node['head'] in byid and node['head'] not in seen:
                seen.add(node['id']);node=byid[node['head']]
            depths.append(len(seen)+1)
        f.update(has_dependency=int(any(t['dep'] for t in tokens)),n_tokens=len(tokens),n_subjects=sum(v for k,v in counts.items() if k.startswith(('nsubj','csubj'))),
                 n_objects=counts['obj']+counts['iobj'],n_clauses=sum(counts[x] for x in ['ccomp','xcomp','advcl','acl','acl:relcl','csubj']),
                 n_subordinate_clauses=sum(counts[x] for x in ['ccomp','advcl','acl','acl:relcl','csubj']),n_coordination=counts['conj'],tree_depth=max(depths,default=0),
                 mean_dep_distance=mean([abs(order[t['id']]-order[t['head']]) for t in tokens if t['head'] in order]),
                 has_verb=int(any(t['pos'] in {'VERB','AUX'} for t in tokens)),pronoun_ratio=sum(t['pos']=='PRON' for t in tokens)/max(1,len(tokens)),target_in_main_clause=0)
        for t in tokens:
            if t['lemma'] not in variants(lemma):continue
            node=t;seen=set();subordinate=False
            while node['id'] not in seen:
                seen.add(node['id'])
                if node['dep'] in {'advcl','acl','acl:relcl','ccomp','xcomp','csubj'}:subordinate=True
                if node['head'] not in byid:break
                node=byid[node['head']]
            if not subordinate:f['target_in_main_clause']=1
        return f


def quality(f):
    # An explicit heuristic baseline, not a learned or certified Coptic GDEX score.
    length=max(0.0,1-abs(f['sent_len']-20)/40)
    return 0.4*length+0.2*f['top5k_ratio']+0.2*(1-min(1,f['pronoun_ratio']))+0.1*f['is_complete_sentence']+0.1*f['target_in_main_clause']
