import sys, json, sqlite3, tempfile, hashlib, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.source_pool import (
    SupplementedCorpus,
    online_examples,
    online_entry,
    online_matches,
    corpus_page,
)
from coptic_examples.ui import online_highlight, search_page


class SourcePoolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "cache").mkdir()
        self.cfg = SimpleNamespace(
            work=self.root, index=self.root / "cache/corpus.sqlite3"
        )

    def test_additive_corpus_preserves_original_database(self):
        for name, word in [
            ("corpus.sqlite3", "base"),
            ("supplemental_corpus.sqlite3", "added"),
        ]:
            db = sqlite3.connect(self.root / "cache" / name)
            db.executescript(
                "CREATE TABLE sentence(id,work,chapter,verse,source,dialect,payload); CREATE TABLE token(sentence_id,lemma,pos); CREATE TABLE metadata(key,value);"
            )
            db.execute(
                "INSERT INTO sentence VALUES(?,?,?,?,?,?,?)",
                (word, "w", "1", "1", "s", "bohairic", "{}"),
            )
            db.execute("INSERT INTO metadata VALUES(?,?)", ("fingerprint", word))
            db.commit()
            db.close()
        before = hashlib.sha256(self.cfg.index.read_bytes()).hexdigest()
        with SupplementedCorpus(self.cfg).connect() as db:
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM sentence").fetchone()[0], 2
            )
            self.assertEqual(
                db.execute(
                    "SELECT value FROM metadata WHERE key='fingerprint'"
                ).fetchone()[0],
                "base:added",
            )
        self.assertEqual(
            before, hashlib.sha256(self.cfg.index.read_bytes()).hexdigest()
        )

    def test_online_source_is_kept_with_dialect_and_markup(self):
        db = sqlite3.connect(self.root / "cache/online_examples.sqlite3")
        db.executescript(
            "CREATE TABLE entry(tla,payload); CREATE TABLE example(id,tla,dialect,ordinal,marked_text,text,translation,title,urn,payload);"
        )
        entry = dict(
            lemma="ⲥⲟⲙⲥ",
            pos="N",
            forms=[dict(orth="ⲥⲟⲙⲥ")],
            senses=[dict(translations=dict(en=["looking"]))],
        )
        db.execute("INSERT INTO entry VALUES(?,?)", ("C3553", json.dumps(entry)))
        db.execute(
            "INSERT INTO example VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                "x",
                "C3553",
                "bohairic",
                1,
                "ⲟⲩ**ⲥⲟⲙⲥ**",
                "ⲟⲩⲥⲟⲙⲥ",
                "look",
                "title",
                "urn",
                "{}",
            ),
        )
        db.execute("CREATE TABLE online_form(form,tla)")
        db.execute("INSERT INTO online_form VALUES(?,?)", ("ⲥⲟⲙⲥ", "C3553"))
        db.commit()
        db.close()
        self.assertEqual([e.tla for e in online_matches(self.cfg, "ⲤⲞⲘⲤ")], ["C3553"])
        self.assertEqual(len(online_examples(self.cfg, "C3553", "bohairic")), 1)
        self.assertEqual(online_examples(self.cfg, "C3553", "sahidic"), [])
        self.assertEqual(online_entry(self.cfg, "C3553").pos, "NOUN")
        self.assertIn("<mark>ⲥⲟⲙⲥ</mark>", online_highlight("ⲟⲩ**ⲥⲟⲙⲥ**"))
        self.assertNotIn("<script>", online_highlight("<script>"))

    def test_raw_candidate_page_keeps_long_examples(self):
        rows = [dict(sent=" ".join(["ⲁ"] * 43)) for _ in range(32)]
        with patch("coptic_examples.sense.native_candidates", return_value=rows):
            page = corpus_page(self.cfg, SimpleNamespace(), "bohairic", 25, 25)
        self.assertEqual(page["total"], 32)
        self.assertEqual(len(page["rows"]), 7)
        self.assertEqual(page["rows"][0]["bound_groups"], 43)

    def test_source_examples_visible_when_model_rejects_everything(self):
        result = dict(
            entry=dict(tla="C3553", lemma="ⲥⲟⲙⲥ", pos="NOUN", gloss="looking"),
            dialect="bohairic",
            ranker="research",
            candidate_count=64,
            results=[],
            manual_examples=[],
            source_examples=[
                dict(
                    marked_text="ⲟⲩ**ⲥⲟⲙⲥ**",
                    translation="look",
                    title="title",
                    urn="urn:test",
                )
            ],
        )
        rendered = search_page(self.cfg, result)
        self.assertIn("Online dictionary examples (1)", rendered)
        self.assertIn("Browse all corpus matches (64)", rendered)
        self.assertIn("No model-selected examples", rendered)
        self.assertNotIn("No eligible examples", rendered)


if __name__ == "__main__":
    unittest.main()
