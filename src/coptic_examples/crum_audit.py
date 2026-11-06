"""Lossless Bible occurrence inventory, then separately auditable interpretation.

Extraction does NOT depend on dialect, TLA resolution or local Bible coverage.
Every explicit {bible:...} occurrence survives with its exact source offsets.
"""

import bisect
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from .config import write_json, database
from .corpus import Corpus
from .dialects import Dialects
from .lexicon import load
from .text import lemma_key, english, norm, variants, location_key

TAG = re.compile(r"⟦([^⟧]*)⟧\{bible:\s*([^}]*)\}")
MARKER = re.compile(r"\{bible\s*:")
PAGE = re.compile(r"(?m)^((?:\d+|[ivxlcdm]+)[ab]?)\s*$")
PARA = re.compile(r"(?m)^¶[^\n]*")
LEAD = re.compile(r"^¶\s*⟨([^⟩]+)⟩")
DIALECT = re.compile(r"⟪([^⟫]+)⟫")
GLOSS = re.compile(r"⟨gloss:\s*([^⟩]+)⟩")
CV = re.compile(r"^(?:NO-LINK:\s*)?(.+?)\s+(\d+):([0-9]+[a-z]?)$")

BOOKS = {
    "Genesis": "gen",
    "Exodus": "exod",
    "Leviticus": "lev",
    "Numbers": "num",
    "Deuteronomy": "deut",
    "Joshua": "josh",
    "Judges": "judg",
    "Ruth": "ruth",
    "First Samuel": "1sam",
    "Second Samuel": "2sam",
    "First Kings": "1kgs",
    "Second Kings": "2kgs",
    "First Chronicles": "1chr",
    "Second Chronicles": "2chr",
    "Ezra": "ezra",
    "Nehemiah": "neh",
    "Esther": "esth",
    "Job": "job",
    "Psalms": "pss",
    "Proverbs": "prov",
    "Ecclesiastes": "eccl",
    "Song of Songs": "song",
    "Song of Solomon": "song",
    "Isaiah": "isa",
    "Jeremiah": "jer",
    "Lamentations of Jeremiah": "lam",
    "Lamentations": "lam",
    "Ezekiel": "ezek",
    "Daniel": "dan",
    "Hosea": "hos",
    "Joel": "joel",
    "Amos": "amos",
    "Obadiah": "obad",
    "Jonah": "jonah",
    "Micah": "mic",
    "Nahum": "nah",
    "Habakkuk": "hab",
    "Zephaniah": "zeph",
    "Haggai": "hag",
    "Zechariah": "zech",
    "Malachi": "mal",
    "Gospel of St. Matthew": "matt",
    "Gospel of St. Mark": "mark",
    "Gospel of St. Luke": "luke",
    "Gospel of St. John": "john",
    "Acts of the Apostles": "acts",
    "Epistle to the Romans": "rom",
    "First Epistle to the Corinthians": "1cor",
    "Second Epistle to the Corinthians": "2cor",
    "Epistle to the Galatians": "gal",
    "Epistle to the Ephesians": "eph",
    "Epistle to the Philippians": "phil",
    "Epistle to the Colossians": "col",
    "First Epistle to the Thessalonians": "1thess",
    "Second Epistle to the Thessalonians": "2thess",
    "First Epistle to Timothy": "1tim",
    "Second Epistle to Timothy": "2tim",
    "Epistle to Titus": "titus",
    "Epistle to Philemon": "phlm",
    "Epistle to the Hebrews": "heb",
    "Epistle of St. James": "jas",
    "First Epistle of St. Peter": "1pet",
    "Second Epistle of St. Peter": "2pet",
    "First Epistle of St. John": "1john",
    "Second Epistle of St. John": "2john",
    "Third Epistle of St. John": "3john",
    "Epistle of St. Jude": "jude",
    "Apocalypse": "rev",
    "Tobit": "tob",
    "Judith": "jdt",
    "Wisdom of Solomon": "wis",
    "Wisdom of Jesus son of Sirach": "sir",
    "Baruch": "bar",
    "Epistle of Jeremiah": "epjer",
    "Susanna": "sus",
    "Bel and the Dragon": "bel",
    "Second Maccabees": "2macc",
    "Prayer of Manasses": "prman",
}


def parse_reference(body):
    body = norm(body)
    m = CV.fullmatch(body)
    if m:
        book, ch, verse = m.groups()
        return dict(
            kind="verse",
            book=book,
            book_key=BOOKS.get(book, ""),
            chapter=location_key(ch),
            verse=location_key(verse),
            no_link=body.startswith("NO-LINK:"),
        )
    clean = re.sub(r"^NO-LINK:\s*", "", body)
    # Lettered additions use their own printed versification. Preserve, do not
    # guess a corresponding numbered Daniel/Esther verse.
    m = re.fullmatch(r"(Daniel|Esther)\s+([A-Z]):(\d+)", clean)
    if m:
        return dict(kind="lettered_section", book=m[1], section=m[2], verse=m[3])
    m = re.fullmatch(r"(Daniel|Esther)\s+([A-Z])", clean)
    if m:
        return dict(kind="lettered_section_only", book=m[1], section=m[2])
    m = re.fullmatch(r"(.+?)\s+(\d+)\s+subscriptio", clean)
    if m:
        return dict(
            kind="subscriptio", book=m[1], book_key=BOOKS.get(m[1], ""), chapter=m[2]
        )
    m = re.fullmatch(r"(.+?)\s+(\d+)(?:\s+(titulus))?", clean)
    if m:
        return dict(
            kind="titulus" if m[3] else "chapter_only",
            book=m[1],
            book_key=BOOKS.get(m[1], ""),
            chapter=m[2],
        )
    if clean in BOOKS:
        return dict(kind="book_only", book=clean, book_key=BOOKS[clean])
    if clean == "first_samuel second_samuel first_kings second_kings":
        return dict(kind="ambiguous_book_group", books=["1sam", "2sam", "1kgs", "2kgs"])
    if (
        clean
        == "first_epistle_to_the_thessalonians second_epistle_to_the_thessalonians"
    ):
        return dict(kind="ambiguous_book_group", books=["1thess", "2thess"])
    return dict(kind="unparsed", body=body)


def page_set(spec):
    pages = set()
    for a, letter, b, last in re.findall(r"(\d+)([ab]?)(?:-(\d+)([ab]?))?", spec):
        start = int(a) * 2 + (letter == "b")
        end = int(b) * 2 + (last == "b") if b else start
        if not letter and not b:
            end = start + 1
        for i in range(start, end + 1):
            pages.add(str(i // 2) + ("b" if i % 2 else "a"))
    for n in re.findall(r"(\d+)ab", spec):
        pages.update([n + "a", n + "b"])
    return pages


def lexical_index(path):
    lex = load(path)
    pages = {}
    ns = "{http://www.w3.org/XML/1998/namespace}"
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag.rsplit("}", 1)[-1] != "entry":
            continue
        found = set()
        for c in el.iter():
            text = norm(c.text)
            if c.tag.rsplit("}", 1)[-1] == "bibl":
                m = re.search(r"\bCD\s+([^;]+)", text)
                if m:
                    found.update(page_set(m[1]))
        pages[el.get(ns + "id", "")] = found
        el.clear()
    byform = defaultdict(list)
    for tla, e in lex.items():
        for form in e.forms:
            byform[form].append(tla)
    return lex, byform, pages


def bind_heading(form, page, gloss, lex, byform, pages):
    key = lemma_key(form)
    pool = byform.get(key, [])
    page_pool = [t for t in pool if page in pages.get(t, set())]
    if not page_pool:
        return [], "unresolved_heading"
    words = set(english(gloss))
    overlaps = {t: len(words & set(english(lex[t].gloss))) for t in page_pool}
    best = max(overlaps.values(), default=0)
    selected = [t for t in page_pool if overlaps[t] == best] if best else page_pool
    return sorted(set(selected)), (
        "exact_form_page_gloss"
        if len(selected) == 1 and best
        else "exact_form_page" if len(selected) == 1 else "ambiguous_form_page"
    )


def heading_forms(para, root):
    end = min(
        [x for x in [para.find("⟨gloss:"), para.find("{bible:")] if x >= 0],
        default=min(len(para), 500),
    )
    forms = []
    for raw in re.findall(r"⟨([^⟩]+)⟩", para[:end]):
        form = lemma_key(raw)
        if not form:
            continue
        forms.append(form)
        if root:
            # Only exact lexicon+page matches can accept an expansion below.
            # Never trim arbitrary suffixes (a source of old false bindings).
            compact = form.replace(" ", "")
            base = root.replace(" ", "")
            if re.search(r"[.\-⸗]\s*$", raw):
                for k in range(1, len(base) + 1):
                    if compact.endswith(base[:k]):
                        forms.append(compact[:-k] + base)
                forms.extend([form + root, form + " " + root])
            if compact in {
                "ⲣ",
                "ⲉⲣ",
                "ϯ",
                "ϫⲓ",
                "ϭⲓ",
                "ⲁⲧ",
                "ⲙⲛⲧ",
                "ϭⲓⲛ",
                "ⲣⲉϥ",
                "ⲙⲁⲛ",
            }:
                forms.extend([form + root, form + " " + root])
    return list(dict.fromkeys(forms))


def extract_file(path, lex, byform, pages):
    text = path.read_bytes().decode("utf-8")
    pg = list(PAGE.finditer(text))
    ps = [m.start() for m in pg]
    paras = list(PARA.finditer(text))
    starts = [m.start() for m in paras]
    matches = list(TAG.finditer(text))
    if len(matches) != len(MARKER.findall(text)):
        raise AssertionError(f"Uncaptured Bible marker: {path}")
    state = dict(
        tlas=[],
        binding="no_heading",
        form="",
        gloss="",
        article=path.name + "#unattached",
        rank=0,
    )
    root_form = ""
    records = []
    for index, p in enumerate(paras):
        end = paras[index + 1].start() if index + 1 < len(paras) else len(text)
        para = text[p.start() : end]
        pageindex = bisect.bisect_right(ps, p.start()) - 1
        page = pg[pageindex][1] if pageindex >= 0 else ""
        lead = LEAD.match(para)
        local_gloss = " | ".join(GLOSS.findall(para))
        if lead and lemma_key(lead[1]):
            form = lead[1]
            separator = (
                "___" in text[paras[index - 1].start() if index else 0 : p.start()]
            )
            if not root_form or separator:
                root_form = lemma_key(form)
            # A new explicit heading never inherits an unrelated previous TLA.
            # Abbreviated derivatives stay unresolved unless independently exact.
            proposed = []
            for candidate_form in heading_forms(para, root_form):
                ts, why = bind_heading(
                    candidate_form, page, local_gloss, lex, byform, pages
                )
                for t in ts:
                    proposed.append(
                        (
                            len(set(english(local_gloss)) & set(english(lex[t].gloss))),
                            t,
                            candidate_form,
                            why,
                        )
                    )
            best = max((x[0] for x in proposed), default=0)
            selected = [x for x in proposed if x[0] == best]
            tlas = sorted({x[1] for x in selected})
            binding = (
                selected[0][3]
                if len(tlas) == 1
                else "ambiguous_form_page" if tlas else "unresolved_heading"
            )
            state = dict(
                tlas=tlas,
                binding=binding,
                form=lemma_key(form),
                gloss=local_gloss,
                article=f"{path.name}#p{index+1}",
                rank=0,
            )
        elif (
            local_gloss
            and root_form
            and re.match(r"^¶\s*[―—–]", para)
            and re.search(r"annotation: (?:noun|verb)", para[:500])
        ):
            tlas, binding = bind_heading(
                root_form, page, local_gloss, lex, byform, pages
            )
            state = dict(
                tlas=tlas,
                binding=binding,
                form=root_form,
                gloss=local_gloss,
                article=f"{path.name}#p{index+1}",
                rank=0,
            )
        local = list(TAG.finditer(para))
        inherited_codes = []
        for j, m in enumerate(local):
            absolute = p.start() + m.start()
            end_offset = p.start() + m.end()
            pageindex = bisect.bisect_right(ps, absolute) - 1
            page = pg[pageindex][1] if pageindex >= 0 else ""
            before = para[local[j - 1].end() if j else 0 : m.start()]
            after = para[
                m.end() : local[j + 1].start() if j + 1 < len(local) else len(para)
            ]
            # Keep all raw dialect evidence. A semicolon ends a post-reference scope.
            after_scope = re.split(r"⟦;⟧|;|\n\s*___", after, maxsplit=1)[0]
            after_tags = DIALECT.findall(after_scope)
            before_tags = DIALECT.findall(before)
            depth = 0
            main_after = []
            for char in after_scope:
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth = max(0, depth - 1)
                elif depth == 0:
                    main_after.append(char)
            primary_tags = DIALECT.findall("".join(main_after))
            if primary_tags:
                codes = after_tags
                scope = "post_reference_evidence"
            elif after_tags:
                codes = list(
                    dict.fromkeys((inherited_codes or before_tags[-1:]) + after_tags)
                )
                scope = "inherited_with_parenthetical_variants"
            elif before_tags:
                codes = before_tags[-1:]
                scope = "preceding_scope_evidence"
            else:
                codes = inherited_codes
                scope = "inherited_paragraph_scope" if codes else "unspecified"
            if codes:
                inherited_codes = codes
            state["rank"] += 1
            oid = hashlib.sha256(
                f"{path.name}\0{absolute}\0{m[0]}".encode()
            ).hexdigest()[:24]
            local_tlas = list(state["tlas"])
            local_binding = state["binding"]
            gender = re.match(r"\s*⟦[fm]⟧\{annotation: (feminine|masculine)\}", after)
            if gender and len(local_tlas) > 1:
                code = "f" if gender[1] == "feminine" else "m"
                selected = [
                    t for t in local_tlas if getattr(lex[t], "genders", ()) == (code,)
                ]
                if selected and len(selected) < len(local_tlas):
                    local_tlas = selected
                    local_binding = (
                        "citation_gender"
                        if len(selected) == 1
                        else "ambiguous_citation_gender"
                    )
            records.append(
                dict(
                    id=oid,
                    file=path.name,
                    file_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    page=page,
                    line=text.count("\n", 0, absolute) + 1,
                    char_start=absolute,
                    char_end=end_offset,
                    byte_start=len(text[:absolute].encode()),
                    raw=m[0],
                    printed_reference=m[1],
                    normalized_reference=m[2],
                    reference=parse_reference(m[2]),
                    article_id=state["article"],
                    section_id=f"{path.name}#p{index+1}",
                    editorial_order=state["rank"],
                    paragraph_reference_order=j + 1,
                    heading_form=state["form"],
                    heading_gloss=local_gloss or state["gloss"],
                    tla_candidates=local_tlas,
                    tla_binding=local_binding,
                    dialect_codes=codes,
                    dialect_scope=scope,
                    dialect_evidence_before=before_tags,
                    dialect_evidence_after=after_tags,
                    paragraph=para.strip(),
                )
            )
    captured = {r["char_start"] for r in records}
    # Includes material before the first ¶. Completeness cannot depend on structure.
    for m in matches:
        if m.start() not in captured:
            records.append(
                dict(
                    id=hashlib.sha256(
                        f"{path.name}\0{m.start()}\0{m[0]}".encode()
                    ).hexdigest()[:24],
                    file=path.name,
                    file_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    page="",
                    line=text.count("\n", 0, m.start()) + 1,
                    char_start=m.start(),
                    char_end=m.end(),
                    byte_start=len(text[: m.start()].encode()),
                    raw=m[0],
                    printed_reference=m[1],
                    normalized_reference=m[2],
                    reference=parse_reference(m[2]),
                    article_id=path.name + "#unattached",
                    section_id=path.name + "#unattached",
                    editorial_order=0,
                    heading_form="",
                    heading_gloss="",
                    tla_candidates=[],
                    tla_binding="outside_paragraph",
                    dialect_codes=[],
                    dialect_scope="unspecified",
                )
            )
    records.sort(key=lambda r: r["char_start"])
    assert len(records) == len(matches) and all(
        text[r["char_start"] : r["char_end"]] == r["raw"] for r in records
    )
    return records


def run(config):
    lex, byform, pages = lexical_index(config.lexicon)
    registry = Dialects()
    corpus = Corpus(config)
    lookup = defaultdict(list)
    with corpus.connect() as db:
        for row in db.execute(
            "SELECT id,work,chapter,verse,dialect FROM sentence WHERE work LIKE 'urn:cts:copticLit:ot.%' OR work LIKE 'urn:cts:copticLit:nt.%'"
        ):
            bits = row["work"].split(":")[-1].split(".")
            if len(bits) > 1:
                lookup[
                    (
                        bits[1].lower(),
                        location_key(row["chapter"]),
                        location_key(row["verse"]),
                    )
                ].append((row["id"], row["dialect"]))
    files = sorted((config.source / "Crum_plain_text").glob("*.txt"))
    records = []
    stats = Counter(files=len(files))
    links = []
    with corpus.connect() as db:
        for fp in files:
            for r in extract_file(fp, lex, byform, pages):
                stats["source_occurrences"] += 1
                stats["ref_" + r["reference"]["kind"]] += 1
                stats["binding_" + r["tla_binding"]] += 1
                ref = r["reference"]
                local = []
                if ref["kind"] == "verse" and ref.get("book_key"):
                    local = lookup.get(
                        (ref["book_key"], ref["chapter"], ref["verse"]), []
                    )
                allowed = registry.from_codes(r["dialect_codes"])
                # Codes for an unregistered dialect are not equivalent to unspecified.
                explicit_other = bool(r["dialect_codes"]) and not allowed
                match_counts = Counter()
                valid_forms = (
                    {lemma_key(r["heading_form"])} if r["heading_form"] else set()
                )
                for tla in r["tla_candidates"]:
                    valid_forms.update(lex[tla].forms)
                target_forms = {v for form in valid_forms for v in variants(form)}
                for sid, dialect in local:
                    if explicit_other or (allowed and dialect not in allowed):
                        continue
                    c = json.loads(
                        db.execute(
                            "SELECT payload FROM sentence WHERE id=?", (sid,)
                        ).fetchone()[0]
                    )
                    hit = any(t["lemma"] in target_forms for t in c["tokens"])
                    if not hit and len(r["tla_candidates"]) == 1:
                        from .sense import align_candidate

                        hit = bool(
                            align_candidate(
                                c, lex[r["tla_candidates"][0]], allow_pos_mismatch=True
                            )
                        )
                    status = (
                        "verified_target_form"
                        if hit
                        else "reference_found_target_unverified"
                    )
                    links.append((r["id"], sid, dialect, status))
                    match_counts[status + ":" + dialect] += 1
                r["local_match_counts"] = dict(match_counts)
                r["link_status"] = (
                    "linked"
                    if match_counts
                    else (
                        "dialect_not_local"
                        if local and (allowed or explicit_other)
                        else (
                            "no_local_verse"
                            if ref["kind"] == "verse"
                            else "non_verse_reference"
                        )
                    )
                )
                stats["link_" + r["link_status"]] += 1
                records.append(r)
    path = config.work / "cache/crum_bible.sqlite3"
    with database(path) as db:
        db.executescript(
            "DROP TABLE IF EXISTS occurrence; DROP TABLE IF EXISTS link; CREATE TABLE occurrence(id TEXT PRIMARY KEY,file TEXT,page TEXT,article TEXT,editorial_order INT,payload TEXT); CREATE TABLE link(occurrence_id TEXT,sentence_id TEXT,dialect TEXT,status TEXT,PRIMARY KEY(occurrence_id,sentence_id)); CREATE INDEX link_status ON link(dialect,status);"
        )
        db.executemany(
            "INSERT INTO occurrence VALUES(?,?,?,?,?,?)",
            [
                (
                    r["id"],
                    r["file"],
                    r["page"],
                    r["article_id"],
                    r["editorial_order"],
                    json.dumps(r, ensure_ascii=False),
                )
                for r in records
            ],
        )
        db.executemany("INSERT OR IGNORE INTO link VALUES(?,?,?,?)", links)
    config.derived.mkdir(exist_ok=True, parents=True)
    with (config.derived / "crum_bible_occurrences.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["raw_marker_count_independent"] = sum(
        len(MARKER.findall(p.read_text())) for p in files
    )
    assert stats["source_occurrences"] == stats["raw_marker_count_independent"]
    stats["extraction_recall_of_explicit_markers"] = 1.0
    stats["verified_links"] = sum(x[3] == "verified_target_form" for x in links)
    stats["training_coverage_is_not_extraction_coverage"] = True
    write_json(config.reports / "crum_bible_audit.json", dict(stats))
    md = [
        "# Crum 全部 Bible 标记提取与匹配审计",
        "",
        "目标是完整保留 Crum 的专业编纂顺序，用于学习他的风格。所有方言、不能定位到节的引用和暂未归入 TLA 的引用都保留；没有先过滤 Sahidic。",
        "",
        "| 检查 | 数量 |",
        "|---|---:|",
    ]
    md += [f"| {k} | {v} |" for k, v in stats.items()]
    md += [
        "",
        "## 完整性保证与正确性边界",
        "",
        "逐文件独立数 `{bible:` 标记；逐条核对字符范围可切回完全相同的原始标记。每条保存文件哈希、行号、字节位置、印刷页、原始引用、规范化引用、词条/段落顺序与方言证据。",
        "这保证当前数字化文本中所有显式 Bible 标记均被提取，不等于证明扫描本所有未标记引用也已由数字化作者正确标注。没有凭空补章节或跨版本节号。",
        "book-only、chapter-only、titulus 和字母章节是有效的引用形态，不应因不能映射一个句子而从提取结果删除。字母章节暂不猜测成 Daniel 的数字章节。",
        "TLA 不唯一时保留候选清单；没有像旧 pick_best 一样静默取第一个。只有目标形式实际出现在对应方言 TT 标注中的链接才标 verified_target_form；仍不是逐 token 专家义项判断。",
        "旧 v15 输出 30,466 条引用，不能代替本次完整原始清单。旧流程还会把尚未识别的新词头继承为上一词条；本次新词头会重置绑定，避免错误继承。",
        "完整清单：`data/derived/crum_bible_occurrences.jsonl`。逐链接：`cache/crum_bible.sqlite3`。待复核状态均可按字段查询，不用从日志猜测。",
    ]
    (config.reports / "CRUM_BIBLE_AUDIT.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    return dict(stats)
