"""Keep recorded editorial selections and partial orders in search results."""

import heapq
from collections import defaultdict


def order(rows):
    known = [r for r in rows if r.get("manual_selected") or r.get("crum_citations")]
    novel = [r for r in rows if r not in known]
    nodes = {r["id"]: r for r in known}
    edges = defaultdict(set)
    degree = {key: 0 for key in nodes}
    groups = defaultdict(list)
    for r in known:
        for cite in r.get("crum_citations", []):
            groups["crum:" + cite["article"]].append((cite["rank"], r["id"]))
        if r.get("manual_priority") is not None:
            groups["tab"].append((r["manual_priority"], r["id"]))
    for group in groups.values():
        levels = sorted({rank for rank, sid in group})
        for a, b in zip(levels, levels[1:]):
            for src in {sid for rank, sid in group if rank == a}:
                for dst in {sid for rank, sid in group if rank == b}:
                    if src != dst and dst not in edges[src]:
                        edges[src].add(dst)
                        degree[dst] += 1
    ready = [
        (-nodes[sid]["shortlist_score"], sid) for sid, d in degree.items() if d == 0
    ]
    heapq.heapify(ready)
    selected = []
    seen = set()
    while ready:
        _, sid = heapq.heappop(ready)
        selected.append(nodes[sid])
        seen.add(sid)
        for nxt in edges[sid]:
            degree[nxt] -= 1
            if degree[nxt] == 0:
                heapq.heappush(ready, (-nodes[nxt]["shortlist_score"], nxt))
    unresolved = [r for r in known if r["id"] not in seen]
    for r in unresolved:
        r["editorial_order_conflict"] = True
    selected += sorted(unresolved, key=lambda r: (-r["shortlist_score"], r["id"]))
    selected += sorted(novel, key=lambda r: (-r["shortlist_score"], r["id"]))
    for i, r in enumerate(selected):
        r["editorial_position"] = i
    return selected
