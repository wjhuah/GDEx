#!/usr/bin/env python3
"""Export every original citation and rebuild augmentation with citation lineage."""

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config, database, write_json
from coptic_examples.crum_audit import MARKER
from coptic_examples.lexicon import load
from coptic_examples.learning import augment
from coptic_examples.inclusion import EntryFeatures, COLUMNS_INCLUSION


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def main():
    cfg = Config()
    out = cfg.derived / "complete_citations"
    out.mkdir(parents=True, exist_ok=True)
    models = {
        str(p): digest(p) for p in (cfg.work / "models").rglob("*") if p.is_file()
    }
    source = {
        p.name: p.read_text() for p in (cfg.source / "Crum_plain_text").glob("*.txt")
    }
    expected = sum(len(MARKER.findall(text)) for text in source.values())
    examples = [json.loads(s) for s in (cfg.derived / "learning_examples.jsonl").open()]
    pair_ids = defaultdict(list)
    for row in examples:
        if row["y"] <= 0:
            continue
        for ref in row.get("crum_refs", []):
            pair_ids[ref["occurrence_id"]].append(
                dict(
                    tla=row["tla"],
                    dialect=row["dialect"],
                    sentence_id=row["candidate"]["id"],
                    text_key=row["text_key"],
                    ranks=row["ranks"],
                )
            )
    links = defaultdict(list)
    with database(cfg.work / "cache/crum_bible.sqlite3") as db:
        for oid, sid, dialect, status in db.execute(
            "SELECT occurrence_id,sentence_id,dialect,status FROM link"
        ):
            links[oid].append(dict(sentence_id=sid, dialect=dialect, status=status))
        occurrences = [
            json.loads(p)
            for (p,) in db.execute(
                "SELECT payload FROM occurrence ORDER BY file,editorial_order,id"
            )
        ]
    assert len(occurrences) == expected
    assert len({r["id"] for r in occurrences}) == expected
    statuses = Counter()
    all_ids = set()
    counts = Counter()
    with (out / "citations.jsonl").open("w") as f:
        for r in occurrences:
            assert source[r["file"]][r["char_start"] : r["char_end"]] == r["raw"]
            assert (
                hashlib.sha256(source[r["file"]].encode()).hexdigest()
                == r["file_sha256"]
            )
            oid = r["id"]
            all_ids.add(oid)
            if pair_ids[oid]:
                status = "training_pair_available"
            elif not r["tla_candidates"]:
                status = "entry_unresolved"
            elif len(r["tla_candidates"]) > 1:
                status = "entry_ambiguous"
            elif r["reference"]["kind"] != "verse":
                status = "reference_requires_expansion"
            elif not links[oid]:
                status = "local_text_unavailable_in_scope"
            else:
                status = "alignment_or_label_review_required"
            statuses[status] += 1
            row = dict(
                r,
                citation_selection_label=1,
                label_unit="original_crum_citation",
                training_status=status,
                training_pairs=pair_ids[oid],
                local_links=links[oid],
            )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    assert sum(statuses.values()) == expected
    report = json.loads((cfg.reports / "learning/comparison.json").read_text())
    lex = load(cfg.lexicon)
    aug_counts = []
    children = Counter()
    fingerprint = digest(out / "citations.jsonl")
    for d in report["results"]:
        dialect = d["dialect"]
        features = EntryFeatures(cfg, lex, dialect)
        data = [r for r in examples if r["dialect"] == dialect]
        for r in data:
            r["x"] = features.vector(r["candidate"], lex[r["tla"]])
        for arch, choice in d["choices"].items():
            setting = choice["setting"]
            ratio = 0 if setting == "none" else float(setting.rsplit("_", 1)[1])
            kind = (
                "hard"
                if setting.startswith("hard")
                else "unrelated" if setting.startswith("unrelated") else "mixed"
            )
            extra = augment(data, ratio, kind, features, lex, {})
            assert len(extra) == choice["added"], (
                dialect,
                arch,
                len(extra),
                choice["added"],
            )
            path = out / (dialect + "_" + arch + "_augmentation.jsonl")
            with path.open("w") as f:
                for r in extra:
                    parents = sorted(
                        {p["occurrence_id"] for p in r.get("crum_refs", [])}
                    )
                    assert all(p in all_ids for p in parents)
                    for p in parents:
                        children[(p, dialect, arch)] += 1
                    row = dict(
                        target_tla=r["tla"],
                        source_tla=r["source_tla"],
                        dialect=dialect,
                        label=-1,
                        derived_label=True,
                        augmentation_type=r["sources"][0],
                        candidate=r["candidate"],
                        source_citation_ids=parents,
                        parent_origin="crum" if parents else "tab",
                        citation_dataset_sha256=fingerprint,
                        x=[
                            (
                                float(v)
                                if v is not None and __import__("math").isfinite(v)
                                else None
                            )
                            for v in r["x"]
                        ],
                    )
                    f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            aug_counts.append(
                dict(
                    dialect=dialect,
                    architecture=arch,
                    setting=setting,
                    rows=len(extra),
                    types=dict(Counter(r["sources"][0] for r in extra)),
                )
            )
        features.close()
    by_parent = defaultdict(list)
    for (parent, dialect, arch), n in children.items():
        by_parent[parent].append(dict(dialect=dialect, architecture=arch, rows=n))
    with (out / "augmentation_coverage.jsonl").open("w") as f:
        for r in occurrences:
            oid = r["id"]
            f.write(
                json.dumps(
                    dict(
                        citation_id=oid,
                        training_pair_count=len(pair_ids[oid]),
                        augmentation_children=by_parent[oid],
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
    assert all(
        digest(Path(p)) == value for p, value in models.items()
    ), "Model files changed"
    write_json(out / "feature_schema.json", dict(columns=COLUMNS_INCLUSION))
    manifest = dict(
        original_citation_records=expected,
        unique_ids=len(all_ids),
        citation_selection_positives=expected,
        training_status=dict(statuses),
        augmentation=aug_counts,
        model_files_unchanged=True,
        files={p.name: digest(p) for p in out.glob("*.jsonl")},
        label_policy="Citation-selection positives preserve every printed marker. Dictionary-conditioned sentence labels require resolved and aligned training pairs. Pending citations remain in the dataset with explicit status.",
    )
    write_json(out / "manifest.json", manifest)
    write_json(cfg.reports / "learning/complete_citations.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
