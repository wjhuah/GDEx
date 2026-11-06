"""Target-centred dependency frames and collocations, counted per dialect."""

import json, math
from collections import Counter, defaultdict
from .config import write_json
from .corpus import Corpus
from .dataset import text_key
from .sense import targets

COLUMNS = [
    "frame_log_frequency",
    "frame_probability",
    "collocation_log_mean",
    "collocation_log_max",
]
CONTENT = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM"}


def events(candidate):
    tokens = candidate["tokens"]
    byid = {t["id"]: t for t in tokens}
    children = defaultdict(list)
    for t in tokens:
        children[t["head"]].append(t)
    out = {}
    for t in tokens:
        if not t["lemma"] or t["pos"] not in CONTENT:
            continue
        roles = sorted(
            ch["dep"]
            for ch in children[t["id"]]
            if ch["dep"]
            in {"nsubj", "obj", "iobj", "obl", "ccomp", "xcomp", "amod", "nmod"}
        )
        frame = (t["lemma"], t["pos"], t["dep"], ",".join(roles))
        pairs = []
        parent = byid.get(t["head"])
        if parent and parent["pos"] in CONTENT:
            pairs.append((t["lemma"], "head:" + t["dep"], parent["lemma"]))
        for ch in children[t["id"]]:
            if ch["pos"] in CONTENT:
                pairs.append((t["lemma"], "child:" + ch["dep"], ch["lemma"]))
        out[t["id"]] = (frame, pairs)
    return out


class Typicality:
    def __init__(self, config, dialect):
        path = config.work / "cache" / ("typicality_" + dialect + ".json")
        data = json.loads(path.read_text())
        self.frames = {tuple(k): v for k, v in data["frames"]}
        self.pairs = {tuple(k): v for k, v in data["pairs"]}
        self.totals = data["totals"]

    def vector(self, candidate, entry):
        ev = events(candidate)
        counts = []
        probabilities = []
        collocations = []
        # Exclude this span's own counts, using the same operation at fit and query time.
        own_frames = Counter(frame for frame, pairs in ev.values())
        own_pairs = Counter(p for frame, pairs in ev.values() for p in pairs)
        own_totals = Counter(frame[0] for frame, pairs in ev.values())
        for i in targets(candidate, entry):
            tid = candidate["tokens"][i]["id"]
            if tid not in ev:
                continue
            frame, pairs = ev[tid]
            count = max(0, self.frames.get(frame, 0) - own_frames[frame])
            total = max(0, self.totals.get(frame[0], 0) - own_totals[frame[0]])
            counts.append(math.log1p(count))
            probabilities.append(count / total if total else 0.0)
            collocations.extend(
                math.log1p(max(0, self.pairs.get(p, 0) - own_pairs[p])) for p in pairs
            )
        return [
            sum(counts) / max(1, len(counts)),
            sum(probabilities) / max(1, len(probabilities)),
            sum(collocations) / max(1, len(collocations)),
            max(collocations, default=0.0),
        ]


def build(config):
    from .dialects import Dialects

    registry = Dialects()
    corpus = Corpus(config)
    counts = defaultdict(
        lambda: dict(frames=Counter(), pairs=Counter(), totals=Counter())
    )
    seen = set()
    for c in corpus.all():
        d = c["dialect_id"]
        if d not in registry.data:
            continue
        key = (d, text_key(c))
        if key in seen:
            continue
        seen.add(key)
        for frame, pairs in events(c).values():
            counts[d]["frames"][frame] += 1
            counts[d]["pairs"].update(pairs)
            counts[d]["totals"][frame[0]] += 1
    for d in registry.data:
        values = counts[d]
        write_json(
            config.work / "cache" / ("typicality_" + d + ".json"),
            dict(
                frames=list(values["frames"].items()),
                pairs=list(values["pairs"].items()),
                totals=dict(values["totals"]),
                deduplicated_spans=sum(k[0] == d for k in seen),
            ),
        )
    return {
        d: dict(frames=len(v["frames"]), pairs=len(v["pairs"]))
        for d, v in counts.items()
    }
