from dataclasses import dataclass
import xml.etree.ElementTree as ET
from .text import norm, lemma_key, pos


@dataclass
class Entry:
    tla: str
    lemma: str
    pos: str
    gloss: str
    forms: tuple
    form_dialects: dict = None
    gloss_source: str = "CCL XML"
    original_gloss: str = ""
    genders: tuple = ()
    subcategories: tuple = ()


def load(path):
    out = {}
    ns = "{http://www.w3.org/XML/1998/namespace}"
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag.rsplit("}", 1)[-1] != "entry":
            continue
        forms, poses, glosses = [], [], []
        form_dialects = {}
        for form in el.iter():
            if form.tag.rsplit("}", 1)[-1] != "form":
                continue
            orths = [
                lemma_key("".join(c.itertext()))
                for c in form.iter()
                if c.tag.rsplit("}", 1)[-1] == "orth"
            ]
            codes = [
                norm("".join(c.itertext()))
                for c in form.iter()
                if c.tag.rsplit("}", 1)[-1] == "usg" and c.get("type") == "geo"
            ]
            for orth in orths:
                form_dialects.setdefault(orth, set()).update(codes)
        for c in el.iter():
            tag = c.tag.rsplit("}", 1)[-1]
            s = norm("".join(c.itertext()))
            if tag == "orth" and lemma_key(s):
                forms.append(lemma_key(s))
            elif tag == "pos" and pos(s):
                poses.append(pos(s))
            elif (
                tag in {"quote", "def"}
                and c.get(ns + "lang", c.get("lang", "")) in {"en", "eng", "english"}
                and s
            ):
                glosses.append(s)
        tla = el.get(ns + "id", "")
        if tla and forms:
            out[tla] = Entry(
                tla,
                forms[0],
                poses[0] if poses else "",
                " | ".join(dict.fromkeys(glosses)),
                tuple(dict.fromkeys(forms)),
                {k: sorted(v) for k, v in form_dialects.items()},
            )
        if tla in out:
            out[tla].subcategories = tuple(
                sorted(
                    {
                        norm(c.text)
                        for c in el.iter()
                        if c.tag.rsplit("}", 1)[-1] == "subc" and norm(c.text)
                    }
                )
            )
            out[tla].genders = tuple(
                sorted(
                    {
                        norm(c.text).rstrip(".")
                        for c in el.iter()
                        if c.tag.rsplit("}", 1)[-1] == "gen" and norm(c.text)
                    }
                )
            )
        el.clear()
    import json
    from pathlib import Path

    supplement = Path(__file__).resolve().parents[2] / "lexical_forms.json"
    if supplement.exists():
        for tla, record in json.loads(supplement.read_text()).items():
            if tla not in out:
                continue
            entry = out[tla]
            for form, codes in record["forms"].items():
                key = lemma_key(form)
                entry.forms = tuple(dict.fromkeys((*entry.forms, key)))
                entry.form_dialects.setdefault(key, [])
                entry.form_dialects[key] = sorted(
                    set(entry.form_dialects[key]) | set(codes)
                )
    from .config import Config, PROJECT

    snapshot = PROJECT / "cache/cdo_index.json"
    if snapshot.exists() and Path(path).resolve() == Config().lexicon.resolve():
        apply_snapshot(out, json.loads(snapshot.read_text()))
    for entry in out.values():
        if "Titel" in entry.subcategories:
            entry.gloss += " (title)"
    return out


def apply_snapshot(entries, records):
    """Merge current definitions and attested dialect forms by TLA identity."""
    for remote in records:
        tla = remote["id"]
        if tla not in entries:
            continue
        entry = entries[tla]
        english = [
            text
            for sense in remote.get("senses", [])
            for text in sense.get("translations", {}).get("en", [])
            if text
        ]
        if english:
            entry.original_gloss = entry.gloss
            entry.gloss = " | ".join(dict.fromkeys(english))
            entry.gloss_source = "https://coptic-dictionary.org/entry/" + tla
        for record in remote.get("forms", []):
            form = lemma_key(record.get("orth", ""))
            code = record.get("dialect", "")
            if not form:
                continue
            entry.forms = tuple(dict.fromkeys((*entry.forms, form)))
            if entry.form_dialects is None:
                entry.form_dialects = {}
            entry.form_dialects[form] = sorted(
                set(entry.form_dialects.get(form, [])) | ({code} if code else set())
            )
    return entries
