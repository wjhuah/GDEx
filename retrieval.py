#!/usr/bin/env python3
"""Corpus-wide Coptic example retrieval and ranking."""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import sqlite3
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Callable, Iterable, Sequence


DEPLOY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_DIR.parent.parent
DICTIONARY_ROOT = PROJECT_ROOT.parent
DEFAULT_CORPUS_ROOT = DICTIONARY_ROOT / "corpora"
DEFAULT_LEXICON = DICTIONARY_ROOT / "Comprehensive_Coptic_Lexicon-v1.2-2020.xml"
DEFAULT_CACHE_DIR = DEPLOY_DIR / "cache"
DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
COPTIC_RE = re.compile(r"[\u03e2-\u03ef\u2c80-\u2cff]")
ATTR_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_.:-]*)\s*=\s*"([^"]*)"')
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "his", "in",
    "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "them", "they", "this", "to", "was", "were", "which", "with",
}
COPTIC_STOPWORDS = {
    "ⲙ", "ⲛ", "ⲉ", "ϩⲛ", "ϩⲓ", "ⲛⲧⲉ", "ⲛⲧⲁ", "ⲡ", "ⲧ",
    "ⲛⲓ", "ⲛⲉ", "ⲟⲩ", "ⲡⲉ", "ⲧⲉ", "ⲁ", "ⲉⲣ", "ⲣ",
    "ϣⲁ", "ⲉⲃⲟⲗ",
}
VALID_UPOS = {
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "SCONJ", "VERB", "X",
}
_MODEL = None


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text).replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_lemma(value: object) -> str:
    return "".join(COPTIC_RE.findall(unicodedata.normalize("NFKC", normalize_text(value))))


def lemma_variants(lemma: str) -> tuple[str, ...]:
    lemma = normalize_lemma(lemma)
    if not lemma:
        return ()
    variants = {lemma}
    variants.add(re.sub(r"([ⲁⲉⲏⲓⲟⲱⲩ])\1", r"\1", lemma))
    for index, char in enumerate(lemma):
        if char in "ⲁⲉⲏⲓⲟⲱⲩ":
            variants.add(lemma[:index] + char + char + lemma[index + 1 :])
    return tuple(sorted(variant for variant in variants if variant))


def normalize_pos(value: object) -> str | None:
    pos = normalize_text(value).casefold().rstrip(".")
    if not pos:
        return None
    direct = {name.casefold(): name for name in VALID_UPOS}
    if pos in direct:
        return direct[pos]
    if pos in {"n", "subst", "substantive", "substantiv", "substantiv?", "noun"}:
        return "NOUN"
    if pos.startswith(("verbalpräfix", "verbalpraefix", "nominalpräfix", "nominalpraefix")):
        return "PART"
    if pos in {"v", "vb"} or pos.startswith(("vb", "verb")):
        return "VERB"
    if pos.startswith(("adj", "adjektiv")):
        return "ADJ"
    if pos.startswith(("adv", "adverb")):
        return "ADV"
    if pos.startswith(("präposition", "praeposition", "präp", "prep", "adposition")):
        return "ADP"
    if pos.startswith(("konj", "conj", "coordinating conjunction")):
        return "CCONJ"
    if pos.startswith(("satzkonverter", "subconj", "subordinating conjunction")):
        return "SCONJ"
    if pos.startswith(("pron", "suffixpron", "poss. pron", "selbst. pers")):
        return "PRON"
    if pos.startswith(("artikel", "article", "art", "det")):
        return "DET"
    if pos.startswith(("zahl", "num", "numeral")):
        return "NUM"
    if pos.startswith(("interjektion", "interj", "intj")):
        return "INTJ"
    if pos.endswith(("artikel", "article")):
        return "DET"
    if pos.startswith(("partikel", "particle", "part", "neg")):
        return "PART"
    if pos in {"a", "n-adj"}:
        return "ADJ"
    if pos.startswith(("p", "pers", "dem.", "poss.")):
        return "PRON"
    return None


@dataclass(frozen=True)
class LexicalEntry:
    tla: str
    lemma: str
    pos: str | None
    gloss: str


class Lexicon:
    def __init__(self, entries: Iterable[LexicalEntry]):
        self.by_tla: dict[str, LexicalEntry] = {}
        self.by_lemma: dict[str, list[LexicalEntry]] = {}
        for entry in entries:
            if not entry.tla or not entry.lemma:
                continue
            self.by_tla.setdefault(entry.tla, entry)
            self.by_lemma.setdefault(entry.lemma, []).append(entry)

    @classmethod
    def load(cls, path: Path) -> "Lexicon":
        if not path.is_file():
            raise FileNotFoundError(f"KELLIA lexicon not found: {path}")
        entries = []
        xml_ns = "{http://www.w3.org/XML/1998/namespace}"
        for _, element in ET.iterparse(path, events=("end",)):
            if element.tag.rsplit("}", 1)[-1] != "entry":
                continue
            tla = element.attrib.get(f"{xml_ns}id", "")
            orthographies = []
            poses = []
            glosses = []
            for child in element.iter():
                tag = child.tag.rsplit("}", 1)[-1]
                text = normalize_text(child.text)
                if tag == "orth" and text:
                    normalized = normalize_lemma(text)
                    if normalized:
                        orthographies.append(normalized)
                elif tag == "pos" and text:
                    mapped = normalize_pos(text)
                    if mapped:
                        poses.append(mapped)
                elif tag in {"quote", "def"} and text:
                    lang = (child.attrib.get(f"{xml_ns}lang")
                            or child.attrib.get("lang", "")).lower()
                    if lang in {"en", "eng", "english"}:
                        glosses.append(text)
            if tla and orthographies:
                lemma = orthographies[0]
                entries.append(LexicalEntry(
                    tla=tla,
                    lemma=lemma,
                    pos=poses[0] if poses else None,
                    gloss=" | ".join(dict.fromkeys(glosses)),
                ))
            element.clear()
        return cls(entries)

    def resolve_tla(self, tla: str) -> LexicalEntry:
        entry = self.by_tla.get(normalize_text(tla).upper())
        if entry is None:
            raise LookupError(f"TLA ID not found in KELLIA lexicon: {tla}")
        if entry.pos is None:
            raise LookupError(f"No supported POS annotation found for {entry.tla}")
        return entry

    def resolve_lemma(self, lemma: str, pos: str, definition: str = "") -> LexicalEntry:
        normalized_lemma = normalize_lemma(lemma)
        normalized_pos = normalize_pos(pos)
        if not normalized_lemma:
            raise ValueError("Enter a Coptic lemma.")
        if normalized_pos is None:
            raise ValueError(f"Unsupported POS: {pos}")
        matches = [
            entry for entry in self.by_lemma.get(normalized_lemma, [])
            if entry.pos == normalized_pos
        ]
        distinct_tlas = {entry.tla for entry in matches}
        if len(distinct_tlas) > 1 and not normalize_text(definition):
            ids = ", ".join(sorted(distinct_tlas))
            raise LookupError(
                f"Ambiguous lemma/POS: {normalized_lemma} ({normalized_pos}) maps to "
                f"{ids}. Use a TLA ID or provide a definition."
            )
        if matches:
            if len(distinct_tlas) > 1 and normalize_text(definition):
                def_tokens = set(
                    re.findall(r"[a-z0-9]+", normalize_text(definition).casefold())
                )
                best_entry = None
                best_score = 0
                for m in matches:
                    gloss_tokens = set(
                        re.findall(r"[a-z0-9]+", normalize_text(m.gloss).casefold())
                    )
                    score = len(def_tokens & gloss_tokens)
                    if score > best_score:
                        best_score = score
                        best_entry = m
                if best_entry is not None and best_score > 0:
                    return LexicalEntry(
                        best_entry.tla,
                        normalized_lemma,
                        normalized_pos,
                        normalize_text(definition) or best_entry.gloss,
                    )
            return LexicalEntry(
                matches[0].tla if len(distinct_tlas) == 1 else "",
                normalized_lemma,
                normalized_pos,
                normalize_text(definition) or matches[0].gloss,
            )
        return LexicalEntry("", normalized_lemma, normalized_pos, normalize_text(definition))


def parse_tt_file(path: Path, corpus_root: Path) -> Iterable[dict]:
    """Yield verse-sized records and token lemma/POS annotations from a Coptic TT file."""
    verse_number = ""
    translation = ""
    groups: list[str] = []
    surfaces: list[str] = []
    tokens: list[dict[str, str]] = []
    sequence = 0
    text = path.read_text(encoding="utf-8")
    has_translation_close = "</translation" in text

    def finish() -> dict | None:
        nonlocal sequence, groups, surfaces, tokens, translation
        sentence = normalize_text(" ".join(groups or surfaces))
        record = None
        if sentence and COPTIC_RE.search(sentence):
            sequence += 1
            record = {
                "sentence_id": f"{path.relative_to(corpus_root).as_posix()}#{verse_number or sequence}:{sequence}",
                "sent": sentence,
                "translation": normalize_text(translation),
                "source": path.relative_to(corpus_root).as_posix(),
                "ref": f"{path.stem}:{verse_number}" if verse_number else path.stem,
                "tokens": tokens.copy(),
            }
        groups = []
        surfaces = []
        tokens = []
        translation = ""
        return record

    for raw_line in StringIO(text):
        line = raw_line.strip()
        if not line:
            continue
        attrs = {key: html.unescape(value) for key, value in ATTR_RE.findall(line)}
        if "verse_n" in attrs or re.search(r"<verse(?:\s|>)", line):
            if groups or surfaces:
                previous = finish()
                if previous:
                    yield previous
            verse_number = attrs.get("verse_n", attrs.get("n", ""))
        if "translation" in attrs:
            translation = attrs["translation"]
        if "norm_group" in attrs:
            groups.append(attrs["norm_group"])

        is_norm_token = bool(re.search(r"<(?:[A-Za-z_][\w.-]*:)?norm(?:\s|>)", line))
        if is_norm_token and "norm_group" not in attrs:
            surface = attrs.get("norm", attrs.get("orig", ""))
            if surface:
                surfaces.append(surface)
            lemma = normalize_lemma(attrs.get("lemma", "") or surface)
            pos = normalize_pos(attrs.get("upos", attrs.get("pos", "")))
            if lemma:
                tokens.append({"lemma": lemma, "pos": pos or ""})

        if "</translation" in line:
            record = finish()
            if record:
                yield record
        elif "</verse" in line and not has_translation_close:
            record = finish()
            if record:
                yield record

    if groups or surfaces:
        record = finish()
        if record:
            yield record


class CorpusIndex:
    """A persistent SQLite lemma/POS index over every .tt file below a corpus root."""

    def __init__(self, corpus_root: Path = DEFAULT_CORPUS_ROOT,
                 database_path: Path = DEFAULT_CACHE_DIR / "corpus.sqlite3"):
        self.corpus_root = corpus_root
        self.database_path = database_path

    def _files_and_fingerprint(self) -> tuple[list[Path], str]:
        if not self.corpus_root.is_dir():
            raise FileNotFoundError(f"Coptic Scriptorium corpus directory not found: {self.corpus_root}")
        files = sorted(path for path in self.corpus_root.rglob("*.tt") if path.is_file())
        if not files:
            raise FileNotFoundError(f"No Coptic Scriptorium .tt files found under {self.corpus_root}")
        digest = hashlib.sha256()
        for path in files:
            stat = path.stat()
            digest.update(
                f"{path.relative_to(self.corpus_root).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
            )
        return files, digest.hexdigest()

    def _is_current(self, fingerprint: str) -> bool:
        if not self.database_path.is_file():
            return False
        with sqlite3.connect(self.database_path) as connection:
            try:
                row = connection.execute(
                    "SELECT value FROM metadata WHERE key = 'corpus_fingerprint'"
                ).fetchone()
            except sqlite3.OperationalError:
                return False
        return bool(row and row[0] == fingerprint)

    def ensure_built(self) -> None:
        files, fingerprint = self._files_and_fingerprint()
        if self._is_current(fingerprint):
            return
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.database_path.with_name(
            f"{self.database_path.stem}.{os.getpid()}.tmp.sqlite3"
        )
        if temporary.exists():
            temporary.unlink()
        try:
            with sqlite3.connect(temporary) as connection:
                connection.execute("PRAGMA journal_mode = DELETE")
                connection.executescript("""
                    CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE sentence (
                        id TEXT PRIMARY KEY,
                        sent TEXT NOT NULL,
                        translation TEXT NOT NULL,
                        source TEXT NOT NULL,
                        ref TEXT NOT NULL,
                        tokens_json TEXT NOT NULL
                    );
                    CREATE TABLE token (
                        sentence_id TEXT NOT NULL,
                        lemma_norm TEXT NOT NULL,
                        pos TEXT NOT NULL,
                        FOREIGN KEY(sentence_id) REFERENCES sentence(id)
                    );
                    CREATE INDEX token_lookup ON token(lemma_norm, pos, sentence_id);
                    CREATE TABLE token_frequency (
                        lemma_norm TEXT PRIMARY KEY,
                        frequency INTEGER NOT NULL
                    );
                """)
                connection.execute("BEGIN")
                sentence_rows = []
                token_rows = []
                sentence_count = 0
                for file_path in files:
                    for record in parse_tt_file(file_path, self.corpus_root):
                        sentence_rows.append((
                            record["sentence_id"], record["sent"], record["translation"],
                            record["source"], record["ref"],
                            json.dumps(record["tokens"], ensure_ascii=False),
                        ))
                        sentence_count += 1
                        for token in record["tokens"]:
                            if token["lemma"]:
                                token_rows.append((
                                    record["sentence_id"], token["lemma"], token["pos"],
                                ))
                        if len(sentence_rows) >= 500:
                            connection.executemany(
                                "INSERT OR IGNORE INTO sentence VALUES (?, ?, ?, ?, ?, ?)",
                                sentence_rows,
                            )
                            connection.executemany(
                                "INSERT INTO token VALUES (?, ?, ?)", token_rows,
                            )
                            sentence_rows.clear()
                            token_rows.clear()
                if sentence_rows:
                    connection.executemany(
                        "INSERT OR IGNORE INTO sentence VALUES (?, ?, ?, ?, ?, ?)",
                        sentence_rows,
                    )
                    connection.executemany("INSERT INTO token VALUES (?, ?, ?)", token_rows)
                connection.execute("""
                    INSERT INTO token_frequency(lemma_norm, frequency)
                    SELECT lemma_norm, COUNT(*) FROM token
                    WHERE lemma_norm != '' GROUP BY lemma_norm
                """)
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES ('corpus_fingerprint', ?)",
                    (fingerprint,),
                )
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES ('sentence_count', ?)",
                    (str(sentence_count),),
                )
            os.replace(temporary, self.database_path)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise

    def search(self, lemma: str, pos: str) -> list[dict]:
        self.ensure_built()
        variants = lemma_variants(lemma)
        if not variants:
            return []
        placeholders = ",".join("?" for _ in variants)
        query = f"""
            SELECT s.id, s.sent, s.translation, s.source, s.ref, s.tokens_json,
                   COUNT(t.sentence_id) AS match_count
            FROM token AS t
            JOIN sentence AS s ON s.id = t.sentence_id
            WHERE t.lemma_norm IN ({placeholders}) AND t.pos = ?
            GROUP BY s.id
            ORDER BY s.source, s.ref, s.id
        """
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, (*variants, pos)).fetchall()
            result = [dict(row) for row in rows]
            terms = {
                token["lemma"]
                for row in result for token in json.loads(row["tokens_json"])
                if token["lemma"]
            }
            if terms:
                term_placeholders = ",".join("?" for _ in terms)
                frequencies = dict(connection.execute(
                    f"SELECT lemma_norm, frequency FROM token_frequency "
                    f"WHERE lemma_norm IN ({term_placeholders})",
                    tuple(terms),
                ).fetchall())
            else:
                frequencies = {}
        for row in result:
            row["tokens"] = json.loads(row.pop("tokens_json"))
            row["match_count"] = int(row["match_count"])
            row["mean_frequency"] = self._mean_frequency(row["tokens"], frequencies)
        return result

    def lemma_pos_counts(self) -> dict[tuple[str, str], int]:
        self.ensure_built()
        with sqlite3.connect(self.database_path) as connection:
            rows = connection.execute("""
                SELECT lemma_norm, pos, COUNT(DISTINCT sentence_id)
                FROM token
                WHERE lemma_norm != '' AND pos != ''
                GROUP BY lemma_norm, pos
            """).fetchall()
        return {(lemma, pos): count for lemma, pos, count in rows}

    @staticmethod
    def _mean_frequency(tokens: list[dict], frequencies: dict[str, int]) -> float:
        words = [
            token["lemma"] for token in tokens
            if token["lemma"] and token["lemma"] not in COPTIC_STOPWORDS
        ]
        return sum(frequencies.get(word, 0) for word in words) / len(words) if words else 0.0


class EmbeddingCache:
    """Persistent JSON vectors, separated into definition and sentence namespaces."""

    def __init__(self, path: Path = DEFAULT_CACHE_DIR / "embeddings.json",
                 model_name: str = DEFAULT_MODEL,
                 model_loader: Callable[[str], object] | None = None):
        self.path = path
        self.model_name = model_name
        self.model_loader = model_loader
        self.data: dict[str, dict[str, list[float]]] = {
            "definitions": {}, "sentences": {},
        }
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("model") == model_name:
                cached = payload.get("embeddings", {})
                self.data["definitions"] = cached.get("definitions", {})
                self.data["sentences"] = cached.get("sentences", {})

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()

    def get_many(self, kind: str, texts: Sequence[str]) -> dict[str, list[float]]:
        if kind not in self.data:
            raise ValueError(f"Invalid embedding cache namespace: {kind}")
        normalized = list(dict.fromkeys(normalize_text(text) for text in texts if normalize_text(text)))
        namespace = self.data[kind]
        missing = [text for text in normalized if self._key(text) not in namespace]
        if missing:
            model = self._get_model()
            encoded = model.encode(
                missing, normalize_embeddings=True, show_progress_bar=False,
            )
            for text, vector in zip(missing, encoded, strict=True):
                values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
                norm = math.sqrt(sum(float(value) ** 2 for value in values))
                if not norm:
                    raise RuntimeError("Embedding model returned an all-zero vector.")
                namespace[self._key(text)] = [float(value) / norm for value in values]
            self._save()
        return {text: namespace[self._key(text)] for text in normalized}

    def _get_model(self):
        global _MODEL
        if _MODEL is None:
            if self.model_loader is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as error:
                    raise RuntimeError(
                        "Install sentence-transformers to compute BGE embeddings "
                        "(pip install -r requirements.txt)."
                    ) from error
                self.model_loader = SentenceTransformer
            _MODEL = self.model_loader(self.model_name)
        return _MODEL

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {"model": self.model_name, "embeddings": self.data}
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match.")
    denominator = math.sqrt(sum(x * x for x in left) * sum(y * y for y in right))
    return sum(x * y for x, y in zip(left, right, strict=True)) / denominator if denominator else 0.0


def _english_tokens(text: str) -> list[str]:
    return [
        token for token in re.findall(r"[a-z0-9]+", normalize_text(text).casefold())
        if token not in STOPWORDS
    ]


def rouge_l_f1(reference: str, candidate: str) -> float:
    left = _english_tokens(reference)
    right = _english_tokens(candidate)
    if not left or not right:
        return 0.0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for column, right_token in enumerate(right, 1):
            if left_token == right_token:
                current.append(previous[column - 1] + 1)
            else:
                current.append(max(previous[column], current[-1]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(right)
    recall = lcs / len(left)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _coarse_score(candidate: dict, definition: str) -> float:
    words = candidate["sent"].split()
    length_score = max(0.0, 1.0 - abs(len(words) - 20) / 40.0)
    punctuation_score = float(bool(re.search(r"[.!?;·…]$", candidate["sent"])))
    translation_score = float(bool(candidate["translation"]))
    lexical_relevance = rouge_l_f1(definition, candidate["translation"])
    return (
        0.60 * lexical_relevance
        + 0.15 * translation_score
        + 0.15 * length_score
        + 0.10 * punctuation_score
    )


class RetrievalEngine:
    def __init__(self, index: CorpusIndex | None = None,
                 embeddings: EmbeddingCache | None = None):
        self.index = index or CorpusIndex()
        self.embeddings = embeddings or EmbeddingCache()

    def search(self, lemma: str, pos: str, definition: str = "",
               top_n: int = 10, candidate_limit: int = 100) -> dict:
        pos = normalize_pos(pos) or ""
        if not pos:
            raise ValueError(f"Unsupported POS: {pos}")
        if not 1 <= top_n <= 50:
            raise ValueError("top_n must be between 1 and 50.")
        if not 1 <= candidate_limit <= 100:
            raise ValueError("candidate_limit must be between 1 and 100.")
        all_candidates = self.index.search(lemma, pos)
        unique = {}
        for candidate in all_candidates:
            key = (
                re.sub(r"\s+", "", normalize_text(candidate["sent"]).casefold()),
                normalize_text(candidate["translation"]).casefold(),
            )
            unique.setdefault(key, candidate)
        candidates = list(unique.values())
        warnings = []
        if not normalize_text(definition):
            warnings.append(
                "No English definition is available; supply one for more reliable semantic ranking."
            )
        if not candidates:
            warnings.append("No corpus sentence matched this lemma and POS.")
        candidates.sort(
            key=lambda item: (
                -_coarse_score(item, normalize_text(definition) or normalize_text(lemma)),
                item["source"], item["ref"], item["sent"],
            )
        )
        prefiltered_count = len(candidates)
        candidates = candidates[:candidate_limit]
        if not candidates:
            return {
                "lemma": lemma, "pos": pos, "definition": definition,
                "candidate_count": 0, "scored_count": 0, "results": [],
                "warnings": warnings,
            }

        candidates = self.score_candidates(candidates, normalize_text(definition) or normalize_text(lemma))
        if len(candidates) < prefiltered_count:
            warnings.append(
                f"Scored the best {len(candidates)} coarse candidates out of "
                f"{prefiltered_count}; candidate_limit is {candidate_limit}."
            )
        if not any(item["rouge_l"] > 0 for item in candidates):
            warnings.append(
                "No scored translation shares content words with the definition; "
                "review these examples as possible low-confidence matches."
            )
        return {
            "lemma": lemma,
            "pos": pos,
            "definition": normalize_text(definition) or normalize_text(lemma),
            "candidate_count": len(all_candidates),
            "deduplicated_count": prefiltered_count,
            "scored_count": len(candidates),
            "candidate_limit": candidate_limit,
            "warnings": warnings,
            "results": candidates[:top_n],
        }

    def score_candidates(self, candidates: Sequence[dict], definition: str) -> list[dict]:
        """Score every supplied candidate, without retrieval caps or deduplication."""
        if not candidates:
            return []
        definition_text = normalize_text(definition)
        if not definition_text:
            raise ValueError("A definition is required to score candidates.")
        scored = [dict(item) for item in candidates]
        sentence_texts = [
            item["translation"] or item["sent"] for item in scored
        ]
        definition_vector = self.embeddings.get_many(
            "definitions", [definition_text],
        )[definition_text]
        sentence_vectors = self.embeddings.get_many("sentences", sentence_texts)
        return self.score_candidates_preembedded(
            scored, definition_text, sentence_vectors, definition_vector,
        )

    def score_candidates_preembedded(
        self, candidates: Sequence[dict], definition: str,
        sentence_vectors: dict[str, list[float]],
        definition_vector: Sequence[float],
    ) -> list[dict]:
        """Score a candidate set using vectors already loaded from EmbeddingCache."""
        if not candidates:
            return []
        definition_text = normalize_text(definition)
        scored = [dict(item) for item in candidates]
        max_frequency = max(
            (math.log1p(max(0.0, float(item.get("mean_frequency", 0) or 0))) for item in scored),
            default=0.0,
        )
        for item in scored:
            sentence_text = item["translation"] or item["sent"]
            if sentence_text not in sentence_vectors:
                raise KeyError(f"Missing precomputed sentence embedding for {sentence_text!r}")
            cosine = (
                cosine_similarity(definition_vector, sentence_vectors[sentence_text])
                if item["translation"] else 0.0
            )
            rouge = rouge_l_f1(definition_text, item["translation"])
            log_frequency = math.log1p(max(0.0, float(item.get("mean_frequency", 0) or 0)))
            frequency_score = log_frequency / max_frequency if max_frequency else 0.0
            item["cosine"] = cosine
            item["rouge_l"] = rouge
            item["frequency_score"] = frequency_score
            item["score"] = 0.55 * cosine + 0.30 * rouge + 0.15 * frequency_score
        scored.sort(
            key=lambda item: (
                -item["score"], -item["cosine"], -item["rouge_l"],
                item.get("source", ""), item.get("ref", ""), item["sent"],
            )
        )
        return scored
