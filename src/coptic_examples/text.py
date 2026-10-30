import re
import unicodedata

COPTIC = re.compile(r"[\u03e2-\u03ef\u2c80-\u2cff]")
ATTR = re.compile(r'''([\w:.-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')''')
STOP = set("a an the of to and or in on at for with by from as is are was were be been being have has had do does did will would shall should may might must can could this that these those it its he she they them his her their him i you we us me my your our but if then so not no than up out into upon about over after before between".split())
UPOS = set("ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ VERB X".split())
# Explicit native mapping from UD Coptic morphology table; raw tags retained.
NATIVE = dict(zip(
    "AAOR ACAUS ACOND ACONJ ADV AFUTCONJ AJUS ALIM ANEGAOR ANEGJUS ANEGOPT ANEGPST ANY AOPT APREC APST ART CCIRC CCOND CFOC CONJ COP CPRET CREL EXIST FM FUT IMOD N NEG NOUN NPROP NUM PDEM PINT PPERI PPERO PPERS PPOS PREP PTC PUNCT UNKNOWN V VBD VIMP VSTAT".split(),
    "AUX AUX SCONJ AUX ADV AUX AUX SCONJ AUX AUX AUX AUX AUX AUX SCONJ AUX DET SCONJ SCONJ PART CCONJ PRON AUX SCONJ VERB X AUX ADV NOUN ADV NOUN PROPN NUM DET PRON PRON PRON PRON DET ADP PART PUNCT X VERB VERB VERB VERB".split()))


def norm(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value or ""))).strip()


def lemma_key(value):
    # Preserve MWE boundaries instead of concatenating separate lexemes.
    return " ".join("".join(COPTIC.findall(x)) for x in norm(value).split()).strip()


def variants(value):
    value = lemma_key(value)
    found = {value, re.sub(r"([ⲁⲉⲏⲓⲟⲱⲩ])\1", r"\1", value)}
    for i, c in enumerate(value):
        if c in "ⲁⲉⲏⲓⲟⲱⲩ":
            found.add(value[:i] + c + value[i:])
    return sorted(x for x in found if x)


def pos(value):
    raw = norm(value)
    if raw.upper() in NATIVE:
        return NATIVE[raw.upper()]
    if raw.upper() in UPOS:
        return raw.upper()
    if "_" in raw and raw.split("_")[-1].upper().startswith("PP"):
        return "PRON"  # documented portmanteau subject handling, retain xpos
    lower = raw.casefold().rstrip(".")
    for prefixes, mapped in [(("subst", "noun"), "NOUN"), (("vb", "verb"), "VERB"),
                             (("adj", "adjektiv"), "ADJ"), (("adv",), "ADV"),
                             (("präposition", "praeposition", "prep"), "ADP"),
                             (("pron",), "PRON"), (("konj",), "CCONJ"),
                             (("artikel",), "DET"), (("zahl",), "NUM")]:
        if lower.startswith(prefixes):
            return mapped
    return ""  # Unknown is measurable, not silently NOUN/VERB.


def english(value):
    return [x for x in re.findall(r"[a-z]+", norm(value).casefold()) if x not in STOP]


def real_translation(value):
    return bool(re.search(r"[A-Za-z]{2,}", norm(value)))


def editorial_heading(value):
    letters=re.sub(r'[^A-Za-z]','',value)
    return bool(len(letters)>20 and letters==letters.upper() and re.search(r'\b(?:MARTYRDOM|LIFE|HOMILY|EPISTLE) OF\b',value))


def sm_freq(gloss, translation):
    g, t = set(english(gloss)), set(english(translation))
    return len(g & t) / len(g) if g else 0.0


def rouge(gloss, translation):
    a, b = english(gloss), english(translation)
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for i, y in enumerate(b, 1):
            cur.append(prev[i-1] + 1 if x == y else max(cur[-1], prev[i]))
        prev = cur
    return 2 * prev[-1] / (len(a) + len(b)) if a and b else 0.0


def work_urn(value):
    # Version aliases seen in manual table vs current Scriptorium; no arbitrary prefix match.
    bits = norm(value).split(":")
    return ":".join(bits[:4]).replace("_ed", "")


def location_key(value):
    return norm(value).lstrip("0") or ("0" if norm(value) else "")
