import json
import tempfile
import unittest
from pathlib import Path

import retrieval
from retrieval import (
    CorpusIndex,
    EmbeddingCache,
    LexicalEntry,
    Lexicon,
    RetrievalEngine,
    normalize_pos,
    parse_tt_file,
    rouge_l_f1,
)
from evaluate_crum4 import label_to_grade, ndcg_at_k


def write_tt(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write('<meta corpus="test">\n')
        for number, (sentence, lemma, pos, translation) in enumerate(rows, 1):
            stream.write(
                f'<verse_n verse_n="{number}" translation="{translation}">\n'
                f'<norm_group norm_group="{sentence}">\n'
                f'<norm lemma="{lemma}" pos="{pos}" norm="{lemma}">\n'
                '</verse_n>\n'
            )


class FakeEmbeddings:
    def get_many(self, kind: str, texts: list[str]) -> dict[str, list[float]]:
        result = {}
        for text in texts:
            lowered = text.casefold()
            result[text] = [1.0, 0.0] if "father" in lowered or "definition" in lowered else [0.0, 1.0]
        return result


class RetrievalTests(unittest.TestCase):
    def test_normalize_pos_maps_kellia_and_corpus_tags(self) -> None:
        self.assertEqual(normalize_pos("Subst."), "NOUN")
        self.assertEqual(normalize_pos("Vb."), "VERB")
        self.assertEqual(normalize_pos("V"), "VERB")
        self.assertEqual(normalize_pos("PREP"), "ADP")

    def test_evaluation_uses_train_v15_grade_thresholds(self) -> None:
        self.assertEqual([label_to_grade(value) for value in (30, 20, 10, 9)], [3, 2, 1, 0])
        self.assertEqual(ndcg_at_k([30, 0], [1.0, 0.0]), (1.0, 1.0))

    def test_tt_parser_yields_lemma_pos_and_translation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "test_TT" / "sample.tt"
            write_tt(source, [("ⲓⲱⲧ", "ⲓⲱⲧ", "N", "A father spoke.")])
            rows = list(parse_tt_file(source, root))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sent"], "ⲓⲱⲧ")
        self.assertEqual(rows[0]["translation"], "A father spoke.")
        self.assertEqual(rows[0]["tokens"], [{"lemma": "ⲓⲱⲧ", "pos": "NOUN"}])

    def test_index_filters_exact_lemma_and_pos_and_engine_ranks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "corpora"
            db = Path(temporary) / "cache" / "corpus.sqlite3"
            write_tt(root / "source_TT" / "examples.tt", [
                ("ⲓⲱⲧ", "ⲓⲱⲧ", "N", "A father spoke."),
                ("ⲓⲱⲧ", "ⲓⲱⲧ", "V", "An unrelated action."),
            ])
            index = CorpusIndex(root, db)
            matches = index.search("ⲓⲱⲧ", "NOUN")
            self.assertEqual(len(matches), 1)
            engine = RetrievalEngine(index=index, embeddings=FakeEmbeddings())
            result = engine.search("ⲓⲱⲧ", "NOUN", "father definition")
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(len(result["results"]), 1)
        self.assertGreaterEqual(result["results"][0]["cosine"], 0.99)
        self.assertGreater(result["results"][0]["rouge_l"], 0)

    def test_embedding_cache_persists_separate_namespaces(self) -> None:
        class FakeModel:
            calls = 0

            def encode(self, texts, **kwargs):
                self.calls += 1
                return [[3.0, 4.0] for _ in texts]

        model = FakeModel()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "embeddings.json"
            cache = EmbeddingCache(path, model_loader=lambda _: model)
            first = cache.get_many("definitions", ["a definition"])
            cache.get_many("sentences", ["a sentence"])
            second = EmbeddingCache(path, model_loader=lambda _: model)
            cached = second.get_many("definitions", ["a definition"])
            stored = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(first, cached)
        self.assertEqual(model.calls, 2)
        self.assertIn("definitions", stored["embeddings"])
        self.assertIn("sentences", stored["embeddings"])
        self.assertAlmostEqual(sum(value * value for value in cached["a definition"]), 1.0)

    def test_rouge_l_and_candidate_limit(self) -> None:
        self.assertEqual(rouge_l_f1("the father spoke", "father spoke"), 1.0)
        engine = RetrievalEngine(index=object(), embeddings=FakeEmbeddings())
        with self.assertRaisesRegex(ValueError, "candidate_limit"):
            engine.search("ⲓⲱⲧ", "NOUN", "father", candidate_limit=101)

    def test_coarse_cap_preserves_gloss_relevant_homograph_candidate(self) -> None:
        class FakeIndex:
            def search(self, lemma, pos):
                candidates = []
                for number in range(10):
                    candidates.append({
                        "id": str(number),
                        "sent": f"ⲉⲓⲱⲧ {number}",
                        "translation": "The father returned.",
                        "source": f"source-{number}",
                        "ref": str(number),
                        "tokens": [],
                        "match_count": 1,
                        "mean_frequency": 1.0,
                    })
                candidates.append({
                    "id": "barley",
                    "sent": "ⲉⲓⲱⲧ ·",
                    "translation": "The barley was harvested.",
                    "source": "z-last-source",
                    "ref": "barley",
                    "tokens": [],
                    "match_count": 1,
                    "mean_frequency": 1.0,
                })
                return candidates

        engine = RetrievalEngine(index=FakeIndex(), embeddings=FakeEmbeddings())
        result = engine.search("ⲉⲓⲱⲧ", "NOUN", "barley", candidate_limit=1)
        self.assertEqual(result["results"][0]["ref"], "barley")
        self.assertTrue(any("coarse candidates" in warning for warning in result["warnings"]))

    def test_missing_definition_is_reported(self) -> None:
        class FakeIndex:
            def search(self, lemma, pos):
                return [{
                    "id": "1",
                    "sent": "ⲓⲱⲧ",
                    "translation": "The word was spoken.",
                    "source": "source",
                    "ref": "1",
                    "tokens": [],
                    "match_count": 1,
                    "mean_frequency": 1.0,
                }]

        result = RetrievalEngine(
            index=FakeIndex(), embeddings=FakeEmbeddings(),
        ).search("ⲓⲱⲧ", "NOUN", "")
        self.assertTrue(any("No English definition" in warning for warning in result["warnings"]))

    def test_lexicon_resolves_tla_and_lemma_pos(self) -> None:
        lexicon = Lexicon([
            LexicalEntry("C1", "ⲓⲱⲧ", "NOUN", "father"),
            LexicalEntry("C2", "ⲓⲱⲧ", "NOUN", "barley"),
            LexicalEntry("C3", "ⲙⲟⲟϣⲉ", "VERB", "walk"),
        ])
        self.assertEqual(lexicon.resolve_tla("c1").pos, "NOUN")
        self.assertEqual(lexicon.resolve_lemma("ⲙⲟⲟϣⲉ", "VERB").tla, "C3")
        with self.assertRaisesRegex(LookupError, "Ambiguous lemma/POS"):
            lexicon.resolve_lemma("ⲓⲱⲧ", "NOUN")
        self.assertEqual(
            lexicon.resolve_lemma("ⲓⲱⲧ", "NOUN", "barley").gloss,
            "barley",
        )


if __name__ == "__main__":
    unittest.main()
