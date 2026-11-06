"""Definition-first admission, before quality or syntactic selection."""

import json, math, re
from collections import defaultdict, Counter
from .config import database
from .text import english, sm_freq, rouge
from .sense import NATIVE, targets

MIN_COSINE = 0.50
MIN_MARGIN = 0.04
MIN_FREQ_MATCH = 0.25


def bleu2(gloss, translation):
    g = english(gloss)
    t = english(translation)
    if not g or not t:
        return 0.0
    scores = []
    for n in range(1, min(2, len(g)) + 1):
        a = Counter(tuple(g[i : i + n]) for i in range(len(g) - n + 1))
        b = Counter(tuple(t[i : i + n]) for i in range(len(t) - n + 1))
        scores.append(sum(min(v, a[k]) for k, v in b.items()) / max(1, sum(b.values())))
    bp = min(1.0, math.exp(1 - len(g) / len(t)))
    return (
        bp * math.exp(sum(math.log(x) for x in scores) / len(scores))
        if all(scores)
        else 0.0
    )


class DefinitionGate:
    def __init__(self, config, lex, entry, dialect):
        from .ablation import CachedVectors

        self.vectors = CachedVectors(config)
        self.entry = entry
        self.dialect = dialect
        self.rivals = [
            e
            for e in lex.values()
            if e.tla != entry.tla
            and e.gloss != entry.gloss
            and set(e.forms) & set(entry.forms)
        ]
        self.citations = defaultdict(list)
        with database(config.work / "cache/crum_bible.sqlite3") as db:
            for payload, sid in db.execute(
                "SELECT o.payload,l.sentence_id FROM occurrence o JOIN link l ON o.id=l.occurrence_id WHERE l.dialect=? AND l.status='verified_target_form'",
                (dialect,),
            ):
                r = json.loads(payload)
                if len(r["tla_candidates"]) == 1:
                    self.citations[sid].append(
                        dict(
                            tla=r["tla_candidates"][0],
                            file=r["file"],
                            line=r["line"],
                            page=r["page"],
                            gloss=r["heading_gloss"],
                            reference=r["normalized_reference"],
                            rank=r["editorial_order"],
                            article=r["article_id"],
                        )
                    )

    def check(self, c, manual_confirmed=False):
        own = self.entry
        ii = targets(c, own)
        from .text import lemma_key

        eligible = [
            e
            for e in self.rivals
            if targets(c, e)
            or any(
                c["tokens"][i]["lemma"] in e.forms
                or lemma_key(c["tokens"][i]["form"]) in e.forms
                for i in ii
            )
        ]
        overlap = sm_freq(own.gloss, c["translation"])
        cosine = self.vectors.cosine(own.gloss, c["translation"])
        competitors = [
            dict(
                tla=e.tla,
                gloss=e.gloss,
                freq_match=sm_freq(e.gloss, c["translation"]),
                cosine=self.vectors.cosine(e.gloss, c["translation"]),
            )
            for e in eligible
        ]
        evidence = self.citations[c["id"]]
        mine = [e for e in evidence if e["tla"] == own.tla]
        others = [e for e in evidence if e["tla"] in {v.tla for v in eligible}]
        status = "review"
        reason = "definition_evidence_pending"
        if manual_confirmed:
            status = "pass"
            reason = "manual_definition_example"
        elif c.get("lexical_condition"):
            status = "pass"
            reason = "crum_listed_construction"
        elif mine and not others:
            status = "pass"
            reason = "crum_definition_citation"
        elif others and not mine:
            reason = "crum_supports_competing_definition"
        elif mine and others:
            reason = "crum_multiple_senses_in_span"
        elif own.tla in NATIVE:
            status = "pass"
            reason = "native_grammatical_definition"
        else:
            max_overlap = max((x["freq_match"] for x in competitors), default=0.0)
            available = [x["cosine"] for x in competitors if x["cosine"] is not None]
            missing = any(x["cosine"] is None for x in competitors)
            if overlap >= MIN_FREQ_MATCH and overlap > max_overlap:
                status = "pass"
                reason = "exclusive_gloss_word_support"
            elif (
                cosine is not None
                and not missing
                and cosine >= MIN_COSINE
                and (not available or cosine - max(available) >= MIN_MARGIN)
            ):
                status = "pass"
                reason = "semantic_threshold_and_margin"
        return dict(
            status=status,
            reason=reason,
            scope="span-level definition evidence; highlighted tokens are TT target occurrences",
            construction=c.get("lexical_condition"),
            freq_match=overlap,
            cosine=cosine,
            rouge_l=rouge(own.gloss, c["translation"]),
            bleu2=bleu2(own.gloss, c["translation"]),
            competing=competitors,
            crum=evidence,
            thresholds=dict(
                min_freq_match=MIN_FREQ_MATCH,
                min_cosine=MIN_COSINE,
                min_cosine_margin=MIN_MARGIN,
            ),
            threshold_status="initial transparent thresholds; expert calibration pending",
        )

    def close(self):
        self.vectors.connection.close()
