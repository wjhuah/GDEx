"""Additive source examples and corpus access; trained inputs stay unchanged."""

import json
import sqlite3
from .corpus import Corpus
from .lexicon import Entry
from .text import lemma_key, pos


class SupplementedCorpus(Corpus):
    retrieval_overlay = True

    def connect(self):
        db = super().connect()
        schemas = ['main']
        for alias, name in [('supplement', 'supplemental_corpus.sqlite3'), ('upstream', 'upstream_corpus.sqlite3')]:
            path = self.config.work / 'cache' / name
            if path.exists():
                db.execute('ATTACH DATABASE ? AS ' + alias, (path.resolve().as_uri()+'?mode=ro',))
                schemas.append(alias)
        if len(schemas) > 1:
            for table in ['sentence', 'token']:
                db.execute('CREATE TEMP VIEW ' + table + ' AS ' + ' UNION ALL '.join('SELECT * FROM '+schema+'.'+table for schema in schemas))
            suffix = ''.join(" || ':' || (SELECT value FROM " + schema + ".metadata WHERE key='fingerprint')" for schema in schemas[1:])
            db.execute("CREATE TEMP VIEW metadata AS SELECT key, CASE WHEN key='fingerprint' THEN value" + suffix + " ELSE value END AS value FROM main.metadata")
        return db


def online_entry(config, tla):
    path = config.work / "cache/online_examples.sqlite3"
    if not path.exists():
        return None
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        row = db.execute("SELECT payload FROM entry WHERE tla=?", (tla,)).fetchone()
    if not row:
        return None
    r = json.loads(row[0])
    forms = tuple(
        dict.fromkeys(
            lemma_key(v["orth"])
            for v in r.get("forms", [])
            if lemma_key(v.get("orth", ""))
        )
    )
    gloss = " | ".join(
        g for s in r.get("senses", []) for g in s.get("translations", {}).get("en", [])
    )
    return Entry(
        tla,
        r.get("lemma", ""),
        pos(r.get("pos", "")),
        gloss,
        forms,
        gloss_source="Online dictionary snapshot",
    )


def online_examples(config, tla, dialect):
    path = config.work / "cache/online_examples.sqlite3"
    if not path.exists():
        return []
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        return [
            dict(r)
            for r in db.execute(
                "SELECT * FROM example WHERE tla=? AND dialect=? ORDER BY ordinal",
                (tla, dialect),
            )
        ]


def corpus_page(config, entry, dialect, offset=0, limit=25):
    from .sense import native_candidates
    from .recommendation_policy import bound_group_count

    if offset < 0 or not 1 <= limit <= 100:
        raise ValueError("Invalid candidate page")
    rows = native_candidates(config, SupplementedCorpus(config), entry, dialect, True)
    return dict(
        total=len(rows),
        offset=offset,
        limit=limit,
        rows=[
            dict(c, bound_groups=bound_group_count(c["sent"]))
            for c in rows[offset : offset + limit]
        ],
    )


def online_matches(config, query):
    path = config.work / "cache/online_examples.sqlite3"
    if not path.exists():
        return []
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        try:
            ids = [
                r[0]
                for r in db.execute(
                    "SELECT DISTINCT tla FROM online_form WHERE form=? ORDER BY tla",
                    (lemma_key(query).casefold(),),
                )
            ]
        except sqlite3.OperationalError:
            return []
    return [entry for tla in ids if (entry := online_entry(config, tla)) is not None]
