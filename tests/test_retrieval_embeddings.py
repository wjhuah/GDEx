import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from coptic_examples.config import Config
from coptic_examples.ablation import CachedVectors
from coptic_examples.embeddings import MODEL


class RetrievalVectorTests(unittest.TestCase):
    def put(self, path, role, text, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path)
        db.execute('CREATE TABLE IF NOT EXISTS vector(model TEXT,role TEXT,key TEXT,payload TEXT,PRIMARY KEY(model,role,key))')
        db.execute('INSERT INTO vector VALUES(?,?,?,?)', (MODEL, role, hashlib.sha256(text.encode()).hexdigest(), json.dumps(value)))
        db.commit()
        db.close()

    def test_original_vectors_take_precedence_and_missing_sentences_use_overlay(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Config(work=Path(d), retrieval_vectors=True)
            extra = cfg.work / 'cache/retrieval_embeddings.sqlite3'
            self.put(cfg.vectors, 'definition', 'cow', [1., 0.])
            self.put(extra, 'definition', 'cow', [0., 1.])
            self.put(extra, 'sentence', 'A cow grazes.', [1., 0.])
            vectors = CachedVectors(cfg)
            try:
                self.assertEqual(vectors.cosine('cow', 'A cow grazes.'), 1.)
                self.assertIsNone(vectors.cosine('cow', 'Uncached sentence.'))
                self.assertIsNone(vectors.get('cow', 'sentence'))
            finally:
                vectors.connection.close()
