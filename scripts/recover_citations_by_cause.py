"""Replay all original citations with independently traceable recovery evidence."""

import bisect
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.citation_recovery import (
    expanded_heading_forms,
    bind_forms,
    token_spans,
    has_lexical_heading,
    has_dialect_alternative,
    explicit_dialect_forms,
    add_cdo_only_entries,
    contextual_dialect_reading,
    explicit_bible_location,
    has_government_alternative,
)
from coptic_examples.config import Config
from coptic_examples.corpus import Corpus
from coptic_examples.crum_audit import (
    lexical_index,
    PARA,
    LEAD,
    GLOSS,
    preceding_dialect_codes,
)
from coptic_examples.dialects import Dialects
from coptic_examples.annotations import load_manual, forbidden_rules, forbidden
from coptic_examples.integrity import alignment_issue
from coptic_examples.sense import NATIVE
from coptic_examples.spelling_compatibility import SpellingCompatibility
from coptic_examples.text import lemma_key, english

cfg = Config()
out = cfg.reports / "本地复盘" / (sys.argv[1] if len(sys.argv) > 1 else "批量修复")
out.mkdir(parents=True, exist_ok=True)
lex, byform, pages = lexical_index(cfg.lexicon)
cdo_records = json.loads((cfg.work / "cache/cdo_index.json").read_text())
cdo_added = add_cdo_only_entries(lex, cdo_records)
for tla in cdo_added:
    pages[tla] = set()
    for form in lex[tla].forms:
        byform[form].append(tla)
(out / "在线新增词条.json").write_text(
    json.dumps(cdo_added, ensure_ascii=False, indent=2) + "\n"
)
registry = Dialects()
exact_spelling = len(sys.argv) > 2 and sys.argv[2] == "exact"
spelling_compatibility = SpellingCompatibility(
    cfg.source / "coptic-nlp/data/norm_table.tab"
)
(out / "异写表来源.json").write_text(
    json.dumps(
        dict(
            file=str(cfg.source / "coptic-nlp/data/norm_table.tab"),
            sha256=spelling_compatibility.sha256,
            enabled=not exact_spelling,
            single_mapping_forms=len(spelling_compatibility.mapping),
            ambiguous_forms_excluded=spelling_compatibility.ambiguous,
            scope="仅原记录代码明确包含S，且语料为sahidic",
        ),
        ensure_ascii=False,
        indent=2,
    )
    + "\n"
)
attested_path = cfg.work / "citation_attested_forms.json"
attested = json.loads(attested_path.read_text()) if attested_path.exists() else {}
with Corpus(cfg).connect() as witness_db:
    for tla, evidence in attested.items():
        for witness in evidence["witnesses"]:
            candidate = json.loads(
                witness_db.execute(
                    "SELECT payload FROM sentence WHERE id=?", (witness["sentence_id"],)
                ).fetchone()[0]
            )
            token = next(
                t for t in candidate["tokens"] if t["id"] == witness["token_id"]
            )
            assert token["form"] == witness["form"] and token["xpos"] == witness["xpos"]
        for form, codes in evidence["forms"].items():
            if form not in lex[tla].forms:
                lex[tla].forms = (*lex[tla].forms, form)
                byform[form].append(tla)
            lex[tla].form_dialects[form] = sorted(
                set(lex[tla].form_dialects.get(form, [])) | set(codes)
            )
rules = forbidden_rules(load_manual(cfg.manual))
counts = Counter()
gains = Counter()
groups = defaultdict(list)
seen = set()
pairs = {}
before = Counter()
comparisons = []


@lru_cache(32)
def article_map(name):
    raw = (cfg.source / "Crum_plain_text" / name).read_bytes()
    text = raw.decode()
    starts, roots, headings = [], [], []
    root = ""
    previous = 0
    for paragraph in PARA.finditer(text):
        if not root or "___" in text[previous : paragraph.start()]:
            lead = LEAD.match(paragraph[0])
            root = lemma_key(lead[1]) if lead else ""
        starts.append(paragraph.start())
        roots.append(root)
        headings.append(paragraph[0])
        previous = paragraph.end()
    return text, hashlib.sha256(raw).hexdigest(), starts, roots, headings


with Corpus(cfg).connect() as db, (out / "逐条修复.jsonl").open("w") as stream:
    location_lookup = defaultdict(list)
    for sid, work, chapter, verse, dialect in db.execute(
        "SELECT id,work,chapter,verse,dialect FROM sentence"
    ):
        key = explicit_bible_location(work, chapter, verse)
        if key and dialect in registry.data:
            location_lookup[key].append(
                dict(sentence_id=sid, dialect=dialect, status="explicit_location_label")
            )

    @lru_cache(2048)
    def sentence(sid):
        return json.loads(
            db.execute("SELECT payload FROM sentence WHERE id=?", (sid,)).fetchone()[0]
        )

    for line in (cfg.derived / "complete_citations/citations.jsonl").open():
        r = json.loads(line)
        oid = r["id"]
        original_dialect_codes = list(r["dialect_codes"])
        active_sahidic = "S" in original_dialect_codes
        if r.get("dialect_scope") == "preceding_scope_evidence":
            before_reference = r["paragraph"].split(r["raw"], 1)[0]
            corrected_codes = preceding_dialect_codes(before_reference)
            if len(corrected_codes) > 1 and r["dialect_codes"] == corrected_codes[-1:]:
                r["dialect_codes"] = corrected_codes
        assert oid not in seen
        seen.add(oid)
        text, sha, starts, roots, headings = article_map(r["file"])
        assert (
            sha == r["file_sha256"]
            and text[r["char_start"] : r["char_end"]] == r["raw"]
        )
        index = bisect.bisect_right(starts, r["char_start"]) - 1
        root = roots[index] if index >= 0 else ""
        # Continuation paragraphs inherit the last explicit lexical heading.
        heading_index = index
        while heading_index >= 0 and not has_lexical_heading(headings[heading_index]):
            if heading_index == 0 or roots[heading_index - 1] != root:
                break
            heading_index -= 1
        paragraph = headings[heading_index] if heading_index >= 0 else ""
        forms = expanded_heading_forms(paragraph, root)
        proposed = bind_forms(forms, r["page"], lex, byform, pages)
        binding_basis = "完整标题与页码"
        if root in forms:
            root_pool = bind_forms([root], r["page"], lex, byform, pages)
            if root_pool:
                proposed = root_pool
                binding_basis = "明确完整母词头优先于变化词形"
        if not proposed and root:
            root_entries = byform.get(root, [])
            root_on_page = any(r["page"] in pages.get(t, set()) for t in root_entries)
            root_spelling = root.replace(" ", "")
            expanded_pool = {
                t
                for form in forms
                if len(root_spelling) >= 3 and root_spelling in form.replace(" ", "")
                for t in byform.get(form, [])
            }
            if root_on_page and len(expanded_pool) == 1:
                proposed = sorted(expanded_pool)
                binding_basis = "完整派生词形与母条目范围；派生条目页码需回看"
        if not proposed:
            # A complete printed heading can identify a lexeme despite coarse pagination.
            full_pool = {
                t
                for form in forms
                if len(form.replace(" ", "")) >= 3
                for t in byform.get(form, [])
            }
            if len(full_pool) == 1:
                proposed = sorted(full_pool)
                binding_basis = "完整标题唯一词条；页码差异已留存"
        if not proposed:
            # Explicit Crum glosses can distinguish same-spelling dictionary entries.
            full_pool = {
                t
                for form in forms
                if len(form.replace(" ", "")) >= 3
                for t in byform.get(form, [])
            }
            initial_context = paragraph.split("{bible:", 1)[0]
            source_words = set(english(" ".join(GLOSS.findall(initial_context))))
            source_words -= {
                "be",
                "do",
                "make",
                "one",
                "in",
                "of",
                "to",
                "a",
                "the",
                "and",
                "or",
                "with",
                "by",
                "as",
                "for",
            }
            scores = {t: source_words & set(english(lex[t].gloss)) for t in full_pool}
            best = max((len(v) for v in scores.values()), default=0)
            winners = [t for t in sorted(full_pool) if len(scores[t]) == best]
            if len(winners) == 1 and (
                best >= 2
                or (best == 1 and any(len(w) >= 5 for w in scores[winners[0]]))
            ):
                proposed = winners
                binding_basis = "完整标题与Crum明确释义消歧；页码差异留存"
        if not proposed and len(root.replace(" ", "")) >= 3:
            header = paragraph.split("⟨gloss:", 1)[0].split("{bible:", 1)[0]
            abbreviations = [
                lemma_key(fragment).replace(" ", "")
                for abbr in re.findall(r"⟨([^⟩]+)⟩", header)
                for fragment in abbr.split(".")[:-1]
            ]
            base_spelling = root.replace(" ", "")
            explicit_root_reference = any(
                any(
                    word.endswith(base_spelling[:n])
                    for n in range(1, len(base_spelling) + 1)
                )
                for word in abbreviations
            )
            root_candidates = [
                t for t in byform.get(root, []) if r["page"] in pages.get(t, set())
            ]
            if explicit_root_reference and root_candidates:
                proposed = root_candidates
                binding_basis = "母条目下的含词根缩写构式；未建立新词条"
        if not proposed and len(root.replace(" ", "")) >= 3:
            # Multiword construction headings can omit the already stated mother lemma.
            lead = LEAD.match(paragraph)
            raw_heading = lemma_key(lead[1]) if lead else ""
            canonical = [
                t
                for t in byform.get(root, [])
                if lex[t].lemma == root and r["page"] in pages.get(t, set())
            ]
            if len(raw_heading.split()) >= 2 and len(canonical) == 1:
                proposed = canonical
                binding_basis = "同一母条目下多词构式，按唯一规范母词头定位"
        if not proposed:
            proposed = list(r["tla_candidates"])
            binding_basis = "原有词条定位"

        # Resolve only dictionary-defined gender contrasts explicitly marked in Crum.
        header = paragraph.split("⟨gloss:", 1)[0].split("{bible:", 1)[0]
        gender = re.findall(r"annotation: (masculine|feminine)", header)
        if len(set(gender)) == 1 and len(proposed) > 1:
            wanted_gender = "m" if gender[0] == "masculine" else "f"
            gender_pool = [
                t
                for t in proposed
                if any(g.rstrip(".") == wanted_gender for g in lex[t].genders)
            ]
            if gender_pool:
                proposed = gender_pool
                binding_basis += "；标题明确语法性别"

        # Explicit same-form noun/verb continuation replaces inherited POS.
        local = r["paragraph"].split(r["raw"], 1)[0]
        grammatical_header = local.split("⟨gloss:", 1)[0].split("{bible:", 1)[0]
        source_parts_of_speech = set(
            re.findall(r"annotation: (noun|verb)\b", grammatical_header)
        )
        if len(source_parts_of_speech) == 1:
            desired = "NOUN" if "noun" in source_parts_of_speech else "VERB"
            matching_pos = [t for t in proposed if lex[t].pos == desired]
            if matching_pos:
                proposed = matching_pos
                binding_basis += "；Crum明确词类"
            else:
                # Continuations can change POS beyond the coarse printed page anchor.
                class_pool = {
                    t
                    for form in [*forms, root]
                    for t in byform.get(form, [])
                    if lex[t].pos == desired
                }
                initial_context = r["paragraph"].split("{bible:", 1)[0]
                source_gloss_words = set(
                    english(" ".join(GLOSS.findall(initial_context)))
                )
                source_gloss_words -= {
                    "be",
                    "do",
                    "make",
                    "one",
                    "in",
                    "of",
                    "to",
                    "a",
                    "the",
                    "and",
                    "or",
                }
                overlaps = {
                    t: len(source_gloss_words & set(english(lex[t].gloss)))
                    for t in class_pool
                }
                best = max(overlaps.values(), default=0)
                winners = [
                    t for t in sorted(class_pool) if best and overlaps[t] == best
                ]
                if len(winners) == 1 or len(class_pool) == 1:
                    proposed = winners if len(winners) == 1 else sorted(class_pool)
                    binding_basis = "Crum词性续段与词典同词形同释义定位；页码差异留存"
        local_links = list(r["local_links"])
        reference = r["reference"]
        if reference.get("kind") == "verse":
            key = (
                reference.get("book_key"),
                reference.get("chapter"),
                reference.get("verse"),
            )
            allowed = registry.from_codes(r["dialect_codes"])
            if not r["dialect_codes"] or allowed:
                known_sids = {link["sentence_id"] for link in local_links}
                for link in location_lookup.get(key, []):
                    if link["sentence_id"] not in known_sids and (
                        not allowed or link["dialect"] in allowed
                    ):
                        local_links.append(link)
                        known_sids.add(link["sentence_id"])
        checks = []
        for link in local_links:
            if not active_sahidic or link["dialect"] != "sahidic":
                continue
            c = sentence(link["sentence_id"])
            codes = registry.data[c["dialect_id"]]["codes"]
            branch_pool = proposed
            branch_basis = binding_basis
            alternatives, alternative_evidence = contextual_dialect_reading(
                r["paragraph"], r["raw"], codes, root
            )
            if not alternatives:
                alternatives = explicit_dialect_forms(
                    r["paragraph"], r["raw"], codes, root
                )
            government_alternative = has_government_alternative(
                r["paragraph"], r["raw"], codes
            )
            if alternatives and not government_alternative:
                alternate_pool = {
                    t for form in alternatives for t in byform.get(form, [])
                }
                if len(alternate_pool) > 1:
                    source_pos = {lex[t].pos for t in proposed}
                    if len(source_pos) == 1:
                        agreeing = []
                        for candidate_tla in alternate_pool:
                            candidate_entry = lex[candidate_tla]
                            candidate_spans = token_spans(
                                c,
                                candidate_entry,
                                normalizer=(
                                    None
                                    if exact_spelling
                                    else spelling_compatibility.key
                                ),
                            )
                            if candidate_entry.pos in source_pos and any(
                                token["pos"] == candidate_entry.pos
                                for token in c["tokens"]
                                if any(token["id"] in span for span in candidate_spans)
                            ):
                                agreeing.append(candidate_tla)
                        if len(agreeing) == 1:
                            alternate_pool = set(agreeing)
                if len(alternate_pool) == 1:
                    branch_pool = sorted(alternate_pool)
                    branch_basis = "本条引文明确记载的方言替换词形"
            for tla in branch_pool:
                e = lex[tla]
                printed_variants = []
                literal_header = paragraph.split("⟨gloss:", 1)[0].split("{bible:", 1)[0]
                literal_forms = [
                    lemma_key(q) for q in re.findall(r"⟨([^⟩]+)⟩", literal_header)
                ]
                if (
                    root in literal_forms
                    and root in e.forms
                    and branch_basis != "本条引文明确记载的方言替换词形"
                ):
                    printed_variants = [
                        f for f in literal_forms if len(f.replace(" ", "")) >= 2
                    ]
                spans = token_spans(
                    c,
                    e,
                    printed_variants,
                    normalizer=None if exact_spelling else spelling_compatibility.key,
                )
                if not spans:
                    continue
                issues = []
                if len(source_parts_of_speech) == 1 and e.pos != desired:
                    issues.append("Crum明确词类与候选词条冲突")
                codes = registry.data[c["dialect_id"]]["codes"]
                if (
                    has_dialect_alternative(r["paragraph"], r["raw"], codes)
                    and branch_basis != "本条引文明确记载的方言替换词形"
                    and not government_alternative
                ):
                    issues.append("当前方言有括注替换读法")
                if forbidden(c, rules[tla]):
                    issues.append("人工标签冲突")
                if alignment_issue(c):
                    issues.append("原译跨度异常")
                pos_matches = any(
                    t["pos"] == e.pos
                    for t in c["tokens"]
                    if any(t["id"] in s for s in spans)
                )
                warnings = []
                if not pos_matches:
                    if (
                        tla in NATIVE
                        or max(len(f.replace(" ", "")) for f in e.forms) < 3
                    ):
                        issues.append("短功能词的原生词性需核对")
                    else:
                        warnings.append("TT词性与词典冲突，保留原标签作为特征")
                allowed = registry.from_codes(r["dialect_codes"])
                if (
                    r["dialect_codes"]
                    and c["dialect_id"] not in allowed
                    and branch_basis != "本条引文明确记载的方言替换词形"
                ):
                    issues.append("方言范围冲突")
                checks.append(
                    dict(
                        tla=tla,
                        gloss=e.gloss,
                        sentence_id=c["id"],
                        dialect=c["dialect_id"],
                        spans=spans,
                        printed_variants=printed_variants,
                        issues=issues,
                        warnings=warnings,
                        binding_basis=branch_basis,
                        alternative_evidence=alternative_evidence,
                        government_alternative=government_alternative,
                        binding_unique=len(branch_pool) == 1,
                        text=c["sent"],
                        translation=c["translation"],
                    )
                )
        matched = {p["tla"] for p in checks}
        accepted = [p for p in checks if not p["issues"] and p["binding_unique"]]
        if not active_sahidic:
            state = "原记录未明确含S，仅记录"
        elif accepted:
            state = "唯一标题与完整词形对齐"
        elif checks:
            state = "词形对齐但词条或标注有冲突"
        elif not proposed:
            state = "标题扩展后仍待定位词条"
        elif not any(link["dialect"] == "sahidic" for link in local_links):
            state = "已有词条线索但缺对应经文"
        else:
            state = "已有词条线索但目标词形不符"
        before[r["training_status"]] += 1
        counts[state] += 1
        has_sahidic_training = any(
            p["dialect"] == "sahidic" for p in r["training_pairs"]
        )
        if not has_sahidic_training and accepted:
            gains[r["training_status"]] += 1
            for p in accepted:
                key = (p["tla"], p["dialect"], p["sentence_id"])
                pair = pairs.setdefault(
                    key,
                    dict(
                        p,
                        citation_ids=[],
                        label=1,
                        label_basis="Crum唯一标题与完整词形自动对齐",
                        expert_verified=False,
                    ),
                )
                pair["citation_ids"].append(oid)
        cause = (
            "缩写或标题继承恢复"
            if set(proposed) != set(r["tla_candidates"])
            else "完整TT词形重新对齐"
        )
        row = dict(
            id=oid,
            file=r["file"],
            page=r["page"],
            reference=r["normalized_reference"],
            original_dialect_codes=original_dialect_codes,
            recovered_dialect_codes=r["dialect_codes"],
            raw=r["raw"],
            root=root,
            forms=forms,
            old_tlas=r["tla_candidates"],
            proposed_tlas=proposed,
            old_status=r["training_status"],
            status=state,
            cause=cause,
            binding_basis=binding_basis,
            expanded_local_links=local_links,
            checks=checks,
            accepted=accepted,
        )
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        groups[state].append(oid)
assert len(seen) == 41808 and sum(counts.values()) == 41808
with (out / "恢复配对_待验证.jsonl").open("w") as f:
    for pair in pairs.values():
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")
report = dict(
    citation_occurrences=len(seen),
    statuses=counts,
    recovered_occurrences_by_previous_status=gains,
    new_staging_pairs=len(pairs),
)
(out / "结果.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
