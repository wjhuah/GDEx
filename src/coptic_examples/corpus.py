"""Index TT translation spans and their token annotations."""

import hashlib
import html
import json
import os
import re
import sqlite3
from collections import Counter
from pathlib import Path

from .text import ATTR, COPTIC, lemma_key, norm, pos, variants, work_urn
from .config import database
from .dialects import Dialects

SCHEMA = "5"


def attrs(line):
    return {m[0]: html.unescape(m[1] or m[2]) for m in ATTR.findall(line)}


def parse_tt(path, root):
    """Verse/translation spans, matching get_examples' citation unit, not guessed sentences.

    Meta translation is a credit, not the example's translation. Native dependency
    arcs and token IDs are retained; spans can include more than one syntactic sentence.
    """
    text = path.read_text(encoding="utf-8-sig")
    groups, tokens, meta = [], [], {}
    chapter = verse = translation = vid = ""
    sequence = 0
    has_translation_close = "</translation>" in text
    registry = Dialects()

    def finish():
        nonlocal groups, tokens, translation, sequence
        sent = norm(" ".join(groups)) or norm(" ".join(t["form"] for t in tokens))
        result = None
        if sent and COPTIC.search(sent):
            sequence += 1
            source = path.relative_to(root).as_posix()
            result = dict(
                id=f"{source}#{sequence}",
                sent=sent,
                translation=norm(translation),
                source=source,
                urn=meta.get("document_cts_urn", ""),
                work=work_urn(meta.get("document_cts_urn", "")),
                chapter=chapter,
                verse=verse,
                vid=vid,
                title=meta.get("title", path.stem),
                dialect=meta.get("languages", meta.get("language", "")),
                segmentation=meta.get("segmentation", ""),
                parsing=meta.get("parsing", ""),
                license=meta.get("license", ""),
                tokens=tokens.copy(),
            )
            result["dialect_id"], result["dialect_evidence"] = registry.identify(
                result["dialect"], source
            )
        groups, tokens, translation = [], [], ""
        return result

    for raw in text.splitlines():
        line = raw.strip()
        a = attrs(line)
        if re.match(r"<meta(?:\s|>)", line):
            meta.update(a)
            chapter = a.get("chapter", chapter)
            continue
        if "chapter_n" in a:
            chapter = a["chapter_n"]
        if "verse_n" in a or re.match(r"<verse(?:\s|>)", line):
            if groups or tokens:
                row = finish()
                if row:
                    yield row
            verse = a.get("verse_n", a.get("n", ""))
            vid = a.get("vid_n", "")
        if "translation" in a:
            translation = a["translation"]
        if "norm_group" in a:
            groups.append(a["norm_group"])
        if re.match(r"<(?:\w+:)?norm(?:\s|>)", line) and "norm_group" not in a:
            rawpos = a.get("pos", "")
            tokens.append(
                dict(
                    id=a.get("xml:id", f"local{len(tokens)+1}"),
                    form=a.get("norm", a.get("orig", "")),
                    lemma=lemma_key(a.get("lemma", a.get("norm", ""))),
                    pos=pos(a.get("upos", rawpos)),
                    xpos=rawpos,
                    dep=a.get("func", a.get("deprel", "")),
                    head=a.get("head", "").lstrip("#"),
                    new_sent=a.get("new_sent", ""),
                )
            )
        if "</translation>" in line or (
            not has_translation_close and "</verse" in line
        ):
            row = finish()
            if row:
                yield row
    row = finish()
    if row:
        yield row


def fingerprint(root):
    files = sorted(Path(root).rglob("*.tt"))
    if not files:
        raise FileNotFoundError(f"No .tt files under {root}")
    digest = hashlib.sha256((SCHEMA + Dialects().fingerprint).encode())
    for p in files:
        s = p.stat()
        digest.update(f"{p.relative_to(root)}\0{s.st_size}\0{s.st_mtime_ns}\n".encode())
    return files, digest.hexdigest()


class Corpus:
    def __init__(self, config):
        self.config = config

    def connect(self):
        db = database(self.config.index.resolve().as_uri() + "?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        return db

    def build(self, force=False):
        files, stamp = fingerprint(self.config.corpus)
        if self.config.index.exists() and not force:
            with self.connect() as db:
                if (
                    db.execute(
                        "SELECT value FROM metadata WHERE key='fingerprint'"
                    ).fetchone()[0]
                    == stamp
                ):
                    return json.loads(
                        db.execute(
                            "SELECT value FROM metadata WHERE key='stats'"
                        ).fetchone()[0]
                    )
        self.config.index.parent.mkdir(parents=True, exist_ok=True)
        temp = self.config.index.with_suffix(".tmp")
        temp.unlink(missing_ok=True)
        stats = Counter(files=len(files))
        with database(temp) as db:
            db.executescript("""
            CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
            CREATE TABLE sentence(id TEXT PRIMARY KEY,work TEXT,chapter TEXT,verse TEXT,source TEXT,dialect TEXT,payload TEXT);
            CREATE TABLE token(sentence_id TEXT,lemma TEXT,pos TEXT);
            CREATE INDEX lookup ON token(lemma,pos,sentence_id);
            CREATE INDEX citation ON sentence(work,chapter,verse);
            CREATE INDEX source_location ON sentence(source,chapter,verse);
            CREATE INDEX sentence_dialect ON sentence(dialect,id);
            CREATE TABLE word_frequency(word TEXT PRIMARY KEY,n INTEGER);
            """)
            frequencies = Counter()
            for i, p in enumerate(files, 1):
                for row in parse_tt(p, self.config.corpus):
                    db.execute(
                        "INSERT INTO sentence VALUES(?,?,?,?,?,?,?)",
                        (
                            row["id"],
                            row["work"],
                            row["chapter"],
                            row["verse"],
                            row["source"],
                            row["dialect_id"],
                            json.dumps(row, ensure_ascii=False),
                        ),
                    )
                    db.executemany(
                        "INSERT INTO token VALUES(?,?,?)",
                        [
                            (row["id"], t["lemma"], t["pos"])
                            for t in row["tokens"]
                            if t["lemma"]
                        ],
                    )
                    # Corpus surface-group counts; fixed for training AND inference.
                    frequencies.update(row["sent"].split())
                    stats["sentences"] += 1
                    stats["tokens"] += len(row["tokens"])
                    stats["unmapped_pos"] += sum(not t["pos"] for t in row["tokens"])
                    stats["dialect_" + row["dialect_id"]] += 1
                if i % 300 == 0:
                    print(f"Indexed {i}/{len(files)} TT files", flush=True)
            db.executemany(
                "INSERT INTO word_frequency VALUES(?,?)", sorted(frequencies.items())
            )
            stats["surface_vocabulary"] = len(frequencies)
            db.executemany(
                "INSERT INTO metadata VALUES(?,?)",
                [
                    ("fingerprint", stamp),
                    ("stats", json.dumps(stats)),
                    ("schema", SCHEMA),
                ],
            )
        os.replace(temp, self.config.index)
        return dict(stats)

    def candidates(self, lemma, upos=None):
        # No top-N cap at retrieval: all matching examples are returned.
        forms = variants(lemma)
        with self.connect() as db:
            if " " in lemma_key(lemma):
                components = [variants(w) for w in lemma_key(lemma).split()]
                possible = self.candidates(lemma.split()[0], upos)
                return [
                    r
                    for r in possible
                    if any(
                        all(
                            t["lemma"] in vs
                            for t, vs in zip(
                                r["tokens"][i : i + len(components)], components
                            )
                        )
                        for i in range(len(r["tokens"]) - len(components) + 1)
                    )
                ]
            if not forms:
                return []
            sql = f"SELECT DISTINCT s.payload FROM sentence s JOIN token t ON t.sentence_id=s.id WHERE t.lemma IN ({','.join('?' for _ in forms)})"
            params = forms[:]
            if upos:
                sql += " AND t.pos=?"
                params += [upos]
            return [
                json.loads(x[0]) for x in db.execute(sql + " ORDER BY s.id", params)
            ]

    def all(self):
        with self.connect() as db:
            for row in db.execute("SELECT payload FROM sentence ORDER BY id"):
                yield json.loads(row[0])

    def citation(self, urn):
        with self.connect() as db:
            return [
                json.loads(x[0])
                for x in db.execute(
                    "SELECT payload FROM sentence WHERE work=? ORDER BY id",
                    (work_urn(urn),),
                )
            ]

    def frequencies(self):
        with self.connect() as db:
            return dict(db.execute("SELECT word,n FROM word_frequency"))
