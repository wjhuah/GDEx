"""Cached BGE sentence embeddings."""

import hashlib
import json
import sqlite3
import numpy as np
from .text import norm
from .config import database

MODEL = "BAAI/bge-large-en-v1.5"


class Embeddings:
    def __init__(self, path, model_name=MODEL, device=None):
        self.path, self.model_name, self.device = path, model_name, device
        self.model = None
        path.parent.mkdir(parents=True, exist_ok=True)
        with database(path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS vector(model TEXT,role TEXT,key TEXT,payload TEXT,PRIMARY KEY(model,role,key))"
            )

    def encode(self, texts, role="sentence"):
        texts = [norm(x) for x in texts]
        unique = sorted(set(texts))
        if not unique:
            return np.empty((0, 1024))
        if "" in unique:
            raise ValueError("Cannot embed an empty text")
        keys = {x: hashlib.sha256(x.encode()).hexdigest() for x in unique}
        found = {}
        with database(self.path) as db:
            for x, k in keys.items():
                row = db.execute(
                    "SELECT payload FROM vector WHERE model=? AND role=? AND key=?",
                    (self.model_name, role, k),
                ).fetchone()
                if row:
                    found[x] = json.loads(row[0])
        missing = [x for x in unique if x not in found]
        if missing:
            if self.model is None:
                from sentence_transformers import SentenceTransformer

                print(f"Loading real {self.model_name}", flush=True)
                self.model = SentenceTransformer(
                    self.model_name, device=self.device, local_files_only=True
                )
            vectors = self.model.encode(
                missing,
                batch_size=16,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=len(missing) > 100,
            )
            if (
                not np.isfinite(vectors).all()
                or (np.linalg.norm(vectors, axis=1) < 0.99).any()
            ):
                raise RuntimeError("Invalid embeddings")
            with database(self.path) as db:
                for x, v in zip(missing, vectors, strict=True):
                    found[x] = v.tolist()
                    db.execute(
                        "INSERT OR REPLACE INTO vector VALUES(?,?,?,?)",
                        (self.model_name, role, keys[x], json.dumps(found[x])),
                    )
        return np.asarray([found[x] for x in texts], dtype=np.float32)
