from dataclasses import dataclass
import xml.etree.ElementTree as ET
from .text import norm, lemma_key, pos


@dataclass
class Entry:
    tla: str
    lemma: str
    pos: str
    gloss: str
    forms: tuple
    form_dialects: dict = None


def load(path):
    out = {}
    ns = '{http://www.w3.org/XML/1998/namespace}'
    for _, el in ET.iterparse(path,events=('end',)):
        if el.tag.rsplit('}',1)[-1] != 'entry':
            continue
        forms, poses, glosses = [], [], []
        form_dialects={}
        for form in el.iter():
            if form.tag.rsplit('}',1)[-1]!='form':continue
            orths=[lemma_key(''.join(c.itertext())) for c in form.iter() if c.tag.rsplit('}',1)[-1]=='orth']
            codes=[norm(''.join(c.itertext())) for c in form.iter() if c.tag.rsplit('}',1)[-1]=='usg' and c.get('type')=='geo']
            for orth in orths:form_dialects.setdefault(orth,set()).update(codes)
        for c in el.iter():
            tag = c.tag.rsplit('}',1)[-1]
            s = norm(''.join(c.itertext()))
            if tag == 'orth' and lemma_key(s):
                forms.append(lemma_key(s))
            elif tag == 'pos' and pos(s):
                poses.append(pos(s))
            elif tag in {'quote','def'} and c.get(ns+'lang',c.get('lang','')) in {'en','eng','english'} and s:
                glosses.append(s)
        tla = el.get(ns+'id','')
        if tla and forms:
            out[tla] = Entry(tla,forms[0],poses[0] if poses else '', ' | '.join(dict.fromkeys(glosses)),tuple(dict.fromkeys(forms)),{k:sorted(v) for k,v in form_dialects.items()})
        el.clear()
    return out
