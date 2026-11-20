import sys, json, csv, re
from pathlib import Path
from collections import Counter

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from coptic_examples.config import Config
from coptic_examples.corpus import Corpus

folder = root / 'reports/local_2218a127/local_40c6713e'
raw = {
    r["id"]: r
    for r in map(
        json.loads, (root / "data/derived/complete_citations/citations.jsonl").open()
    )
}
rows = list(map(json.loads, (folder / '').open()))
exact = {
    r["id"]: r
    for r in map(json.loads, (folder.parent / 'local_23cf2822/local_b415e7f6').open())
}
pairs = {}
gains = Counter()
for r in rows:
    if (
        not any(p["dialect"] == "sahidic" for p in raw[r["id"]]["training_pairs"])
        and r["accepted"]
    ):
        gains[r["old_status"]] += 1
        for p in r["accepted"]:
            key = (p["tla"], p["dialect"], p["sentence_id"])
            pair = pairs.setdefault(
                key,
                dict(
                    p,
                    citation_ids=[],
                    label=1,
                    label_basis='',
                    expert_verified=False,
                ),
            )
            pair["citation_ids"].append(r["id"])
(folder / '').write_text(
    "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs.values())
)
result = json.loads((folder / '').read_text())
result["recovered_occurrences_by_previous_status"] = gains
result["new_staging_pairs"] = len(pairs)
(folder / '').write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n"
)
new = [r for r in rows if r["accepted"] and not exact[r["id"]]["accepted"]]
lost = [r for r in rows if not r["accepted"] and exact[r["id"]]["accepted"]]
(folder / '').write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new)
)
(folder / '').write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lost)
)
(folder / '').write_text(
    json.dumps(
        dict(newly_aligned=len(new), new_conflict=len(lost), net=len(new) - len(lost)),
        ensure_ascii=False,
        indent=2,
    )
    + "\n"
)
reasons = {
    '': '',
    '': '',
    '': '',
    '': '',
}
with Corpus(Config()).connect() as db, (folder / '').open("w") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(
        [
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
        ]
    )
    n = 0
    for r in rows:
        if (
            "S" not in r["original_dialect_codes"]
            or any(p["dialect"] == "sahidic" for p in raw[r["id"]]["training_pairs"])
            or r["accepted"]
        ):
            continue
        coptic = []
        positions = []
        for link in r["expanded_local_links"]:
            if link["dialect"] != "sahidic":
                continue
            item = db.execute(
                "SELECT payload FROM sentence WHERE id=?", (link["sentence_id"],)
            ).fetchone()
            if item:
                coptic.append(json.loads(item[0])["sent"])
                positions.append(link["sentence_id"])
        source = raw[r["id"]]
        quotes = [
            q
            for q in re.findall("⟨([^⟩]+)⟩", source["paragraph"])
            if re.search("[\u03e2-\u03ef\u2c80-\u2cff]", q)
        ]
        writer.writerow(
            [
                r["id"],
                r["file"],
                source["char_start"],
                ",".join(r["original_dialect_codes"]),
                " / ".join(r["forms"]),
                ",".join(r["proposed_tlas"]),
                r["status"],
                reasons[r["status"]],
                " / ".join(quotes),
                " || ".join(dict.fromkeys(coptic)),
                " || ".join(dict.fromkeys(positions)),
            ]
        )
        n += 1
    assert (
        n
        == json.loads((folder / '').read_text())["exclusive_statuses"][
            ''
        ]
    ), n
print(
    "scoped staging",
    len(pairs),
    "pending examples",
    n,
    "new",
    len(new),
    "conflict",
    len(lost),
)
