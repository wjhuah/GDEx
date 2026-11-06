#!/usr/bin/env python3
"""Save the CDO entry index used by the published dictionary."""

import hashlib, json, sys, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coptic_examples.config import Config, write_json

cfg = Config()
url = "https://coptic-dictionary.org/index.json"
raw = urllib.request.urlopen(url, timeout=60).read()
data = json.loads(raw)
if not isinstance(data, list) or not all("id" in r and "senses" in r for r in data):
    raise ValueError("Unexpected CDO index schema")
p = cfg.work / "cache/cdo_index.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_bytes(raw)
write_json(
    cfg.work / "cache/cdo_index_source.json",
    dict(url=url, sha256=hashlib.sha256(raw).hexdigest(), entries=len(data)),
)
print(len(data), "CDO entries saved")
