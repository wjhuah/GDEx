from dataclasses import dataclass
from pathlib import Path
import os
import sqlite3

PROJECT = Path(__file__).resolve().parents[2]


class ManagedConnection(sqlite3.Connection):
    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            self.close()


def database(path, **kwargs):
    return sqlite3.connect(path, factory=ManagedConnection, **kwargs)


@dataclass(frozen=True)
class Config:
    # All original inputs remain read-only outside this new project.
    source: Path = Path(
        os.environ.get("COPTIC_SOURCE_ROOT", str(PROJECT.parent))
    ).resolve()
    work: Path = PROJECT

    @property
    def corpus(self):
        return self.source / "corpora"

    @property
    def lexicon(self):
        return self.source / "Comprehensive_Coptic_Lexicon-v1.2-2020.xml"

    @property
    def manual(self):
        return self.source / "citations_manual_merged.tab"

    @property
    def index(self):
        return self.work / "cache/corpus.sqlite3"

    @property
    def vectors(self):
        return self.work / "cache/embeddings.sqlite3"

    @property
    def derived(self):
        return self.work / "data/derived"

    @property
    def reports(self):
        return self.work / "reports"


def write_json(path, value):
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
