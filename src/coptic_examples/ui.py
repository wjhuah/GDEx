"""HTML views for dictionary examples and evaluation."""

import html
import json
import re
from urllib.parse import urlencode
from .dialects import Dialects
from .selection import choose, metrics

NAMES = {
    "quality_first_joint": "Quality-first + joint ranker",
    "two_head_baseline": "Dual-task baseline (re-evaluated)",
    "random": "Random",
    "gdex_rule": "Fixed GDEX heuristic",
    "tree_gdex": "GDEX only",
    "tree_ud": "UD only",
    "tree_gdex_ud": "GDEX + UD",
    "tree_full_no_sense": "GDEX + UD + information",
    "tree_full": "Full 45-feature ranker",
    "tree_full_no_gdex": "Without GDEX",
    "tree_full_no_ud": "Without UD",
    "tree_full_no_info": "Without information features",
    "tree_extended_ud": "With extended UD features",
    "tree_tab_only": "TAB-only training",
    "tree_crum_only": "Crum-only training",
    "tree_321_floor": "Legacy Crum label floor",
    "tree_naive_homograph": "+ Naive homograph negatives",
    "tree_barley_father": "+ barley/father negatives",
    "crum4_original": "Original Crum4 formula",
    "tree_cached_cheap": "Same subset: feature-based ranker",
    "tree_cached_bge": "Same subset: ranker + BGE",
    "tree_cached_bge_barley": "Same subset: ranker + BGE + barley/father negatives",
}
NAMES.update(
    {
        "tree_no_sent_len": "Without sentence length",
        "tree_no_length": "Without all length features",
    }
)
esc = html.escape


def shell(body, title="Examples", active="search"):
    links = [
        ("search", "/", "Examples"),
        ("sense", "/sense-review", "Sense audit"),
    ]
    nav = "".join(
        f"<a"
        + (' class="active"' if key == active else "")
        + f' href="{url}">{label}</a>'
        for key, url, label in links
    )

    return f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)} · Coptic Example Recommendation</title>
<link rel="stylesheet" href="/review.css">

<div class="site">
    <header>
        <h1>Coptic Example Finder</h1>
        <span class="coptic wordmark">Ⲗⲉⲝⲓⲕⲟⲛ ⲛϢⲛⲉ</span>
    </header>

    <nav aria-label="Main navigation">
        {nav}
        <a href="https://coptic-dictionary.org/"
           target="_blank" rel="noopener">
            Coptic Dictionary ↗
        </a>
    </nav>

    <main>{body}</main>

    <footer>
        <a href="https://coptic-dictionary.org/">
            Coptic Dictionary Online
        </a>
        ·
        <a href="/experiments">Evaluation</a>
    </footer>
</div>
</html>"""


def search_form(
    result=None, query_value="", dialect_value="sahidic", ranker_value="research"
):
    dialect = result.get("dialect", "sahidic") if result else dialect_value
    query = result["entry"]["tla"] if result else query_value
    ranker = (
        result.get("requested_ranker", result.get("ranker", "research"))
        if result
        else ranker_value
    )
    options = "".join(
        f'<option value="{esc(k)}"'
        + (" selected" if k == dialect else "")
        + f'>{esc(v["name"])}</option>'
        for k, v in Dialects().data.items()
    )
    methods = {
        "research": "Learned inclusion (TAB + Crum)",
        "definition_syntax": "Rule-gated syntax baseline",
        "cheap": "Dual-task Baseline",
        "quality_first": "Quality-first Baseline",
    }
    rankers = "".join(
        f'<option value="{k}"' + (" selected" if k == ranker else "") + f">{v}</option>"
        for k, v in methods.items()
    )
    return f"""<aside class="panel search-panel"><h2>Find Examples</h2><form action="/search" method="get">
<label for="query">TLA ID or Lemma</label><input id="query" name="query" value="{esc(query)}" required placeholder="TLA ID or Coptic lemma">
<label for="dialect">Dialect</label><select id="dialect" name="dialect">{options}</select>
<label for="ranker">Ranking Method</label><select id="ranker" name="ranker">{rankers}</select><button type="submit">Search</button></form>
</aside>"""


def highlighted(text, entry, candidate=None):
    if candidate is None:
        return esc(text)
    from .token_highlight import render

    evidence = candidate.get("target_token_evidence")
    if evidence is None:
        from types import SimpleNamespace
        from .sense import targets

        evidence = [
            candidate["tokens"][i] for i in targets(candidate, SimpleNamespace(**entry))
        ]
    return render(text, candidate["tokens"], [t["id"] for t in evidence])


def syntax_label(signature):
    roles = {
        "nsubj": "Subject",
        "obj": "Object",
        "iobj": "Indirect object",
        "nmod": "Nominal modifier",
        "obl": "Oblique",
        "root": "Root",
        "conj": "Conjunct",
        "appos": "Apposition",
    }
    labels = []
    for s in signature:
        if s.startswith("role:"):
            labels.append(roles.get(s[5:], s[5:]))
        elif s == "clause:subordinate":
            labels.append("Subordinate clause")
        elif s == "clause:matrix_or_fragment":
            labels.append("Main clause or fragment")
        elif s.startswith("modifier:"):
            labels.append("Target has modifier")
    return " · ".join(dict.fromkeys(labels)) or "Syntax not verified"


def citation(c):
    title = c.get("title") or c.get("work") or c.get("urn", "")
    return (
        esc(str(title))
        + f" · {esc(str(c.get('chapter','')))}:{esc(str(c.get('verse','')))}"
    )


def search_page(
    config,
    result=None,
    error="",
    choices=None,
    query="",
    dialect="sahidic",
    ranker="research",
):
    content = ""
    if choices:
        content = '<section class="panel"><h2>Choose an entry</h2><table class="entry-choices"><thead><tr><th>Entry</th><th>Forms</th><th>Meaning</th></tr></thead><tbody>'
        for entry in choices:
            url = "/search?" + urlencode(
                dict(query=entry["tla"], dialect=dialect, ranker=ranker)
            )
            content += (
                '<tr><td><a class="coptic" href="'
                + esc(url)
                + '">'
                + esc(entry["lemma"])
                + '</a><br><a href="'
                + esc(url)
                + '">'
                + esc(entry["tla"])
                + "</a>"
                + " · "
                + esc(entry["pos"])
                + "</td><td class=coptic>"
                + esc(", ".join(f for f in entry["forms"] if f != entry["lemma"]))
                + "</td><td>"
                + esc(entry["gloss"])
                + "</td></tr>"
            )
        content += "</tbody></table></section>"
    elif error:
        content = f'<section class="panel"><h2>Search</h2><p class="error">{esc(error)}</p></section>'
    if result:
        e = result["entry"]
        rows = result["results"]
        short = result.get("shortlist_evidence", {})
        provisional = short.get("model_status", {}).get("provisional", False)
        content = (
            f'<section class="panel entry"><div class="entry-top"><h2><span class="coptic lemma">{esc(e["lemma"])}</span> <span class="pos">{esc(e.get("pos") or "")}</span></h2><span class="dialect">{esc(Dialects().data[result["dialect"]]["name"])}</span></div><p class="definition">{esc(e["gloss"])}</p><p class="muted">{esc(e["tla"])} · Raw matches:  {result["candidate_count"]:,} · Candidates: {len(rows)} · '
            + (
                "Learned inclusion · TAB + Crum"
                if result["ranker"] == "research"
                else (
                    "Rule-gated syntax baseline"
                    if result["ranker"] == "definition_syntax"
                    else (
                        "Quality-first"
                        if result["ranker"] == "quality_first"
                        else "Dual-task Baseline"
                    )
                )
            )
            + "</p></section>"
        )
        if not rows:
            content += '<section class="panel"><h2>No eligible examples</h2><p>Try another entry or dialect; results are not mixed across dialects.</p></section>'
        else:
            selected = choose(rows, 3, "coverage")
            direct = choose(rows, 3, "rank")
            covered = metrics(selected, rows)
            ranked = metrics(direct, rows)
            ids = {r["id"]: i for i, r in enumerate(rows, 1)}
            names = lambda xs: ", ".join("#" + str(ids[r["id"]]) for r in xs)
            content += f"""<section class="panel selection"><h3>Three-example Sets <span class="small-tag">Comparison</span></h3><div class="set-compare"><div><b>Top three: {names(direct)}</b><p>Syntax coverage {ranked['weighted_syntax_coverage']:.0%} · Mean selection score {ranked['mean_selection_score']:.3f}</p></div><div><b>Order + coverage: {names(selected)}</b><p>Syntax coverage {covered['weighted_syntax_coverage']:.0%} · Mean selection score {covered['mean_selection_score']:.3f}</p></div></div></section>"""
            content += '<section class="panel examples"><h3>Candidate Examples</h3><ol class="example-list">'
            selected_ids = {r["id"] for r in selected}
            for i, c in enumerate(rows, 1):
                badge = (
                    '<span class="small-tag">Selected for coverage</span>'
                    if c["id"] in selected_ids
                    else ""
                )
                sig = c.get("syntax", {}).get("signature", [])
                rel = c.get("definition_relevance", {})
                content += f"""<li id="example-{i}"><div class="example-heading"><b>#{i}</b> {badge}<span class="score">{'Recommendation score' if result['ranker']=='research' else 'Syntax order score' if result['ranker']=='definition_syntax' else 'Rank score'} {c['score']:.3f}</span></div>
<p class="coptic sentence">{highlighted(c['sent'],e,c)}</p><p class=syntax>TT target tokens: {esc(' | '.join(t['form']+' / '+t['lemma']+' / '+t['xpos']+' / '+t['id'] for t in c.get('target_token_evidence',[])))} — segmentation={esc(c.get('segmentation','unspecified'))}, parsing={esc(c.get('parsing','unspecified'))}</p><p class="translation">{esc(c['translation'])}</p><p class="source">{citation(c)}</p><p class=syntax>{'Crum citation' if c.get('crum_citations') else 'TAB example' if c.get('manual_selected') else ''}</p><p class=syntax>{esc(str((rel.get('construction') or {}).get('display','')))}</p><p class="syntax">{esc(syntax_label(sig))}</p>
<details><summary>Ranking details</summary><div class="evidence"><p>Baseline inclusion <b>{c.get('acceptance_score',c.get('quality_score',0)):.3f}</b> · Legacy full-feature style diagnostic <b>{c.get('crum_style_percentile',0):.0%}</b> · Information <b>{c.get('information_score',0):.3f}</b></p>
<p>Inclusion model: {c.get('learned_inclusion_score',0):.3f}</p><p>Crum references: {esc(str(c.get('crum_citations',[])))}</p><p>Gloss overlap {rel.get('freq_match',0):.3f} · cosine {esc(str(rel.get('cosine')))} · ROUGE-L {rel.get('rouge_l',0):.3f}</p><p>POS alignment review: {esc(str(c.get('target_alignment') or c.get('crum_pos_alignment') or 'lemma and native tags'))}</p><p>Syntactic signature: {esc(', '.join(sig))}</p><p class="muted">Corpus: {esc(c.get('source',''))}<br>Reference: {esc(c.get('urn',''))}</p></div></details></li>"""
            content += "</ol></section>"
        review = short.get("definition_review", [])
        if review:
            content += (
                "<details class=panel><summary>Definition review candidates</summary>"
            )
            for c in review:
                rel = c["definition_relevance"]
                content += (
                    "<article><p class=coptic>"
                    + highlighted(c["sent"], e, c)
                    + "</p><p>"
                    + esc(c["translation"])
                    + "</p><p>"
                    + esc(rel["reason"])
                    + " · Crum: "
                    + esc(str(rel["crum"]))
                    + "</p></article>"
                )
            content += "</details>"
        manual = result.get("manual_examples", [])
        content += f'<details class="panel"><summary>Original TAB examples ({len(manual)}; current dialect)</summary>'
        for r in manual:
            c = r["candidate"]
            content += f'<article class="manual-example"><b>Priority {r.get("priority") or "Unranked"}</b><p class="coptic sentence">{highlighted(c["sent"],e,c)}</p><p>{esc(c["translation"])}</p><p class="source">{citation(c)}</p></article>'
        if not manual:
            content += "<p>No aligned TAB examples for this dialect.</p>"
        content += "</details>"
        excluded = short.get("excluded", result.get("excluded", {}))
        content += (
            '<details class="panel"><summary>Retrieval details</summary><p>Excluded: '
            + esc(", ".join(f"{k}: {v}" for k, v in excluded.items()))
            + "</p></details>"
        )
    return shell(
        '<div class="workspace">'
        + search_form(result, query, dialect, ranker)
        + '<div class="results">'
        + content
        + "</div></div>"
    )


def guide_page(config):
    path = config.work / "docs/USER_GUIDE.md"
    source = (
        path.read_text()
        if path.exists()
        else "# User guide\n\n## Search\nEnter a TLA ID or Coptic lemma, choose a dialect, then select an entry.\n\n## Results\nCrum and TAB selections appear first. Ranking details show model scores and source references."
    )
    parts = []
    paragraph = []
    step = 0

    def flush():
        if paragraph:
            parts.append("<p>" + esc(" ".join(paragraph)) + "</p>")
            paragraph.clear()

    for line in source.splitlines():
        if line.startswith("# "):
            flush()
            parts.append("<h2>" + esc(line[2:]) + "</h2>")
        elif line.startswith("## "):
            flush()
            step += 1
            parts.append(f'<h3 id="step{step}">' + esc(line[3:]) + "</h3>")
        elif not line.strip():
            flush()
        else:
            paragraph.append(line)
    flush()
    toc = (
        '<div class="toc">'
        + "".join(
            f'<a href="#step{i}">{i}. {name}</a>'
            for i, name in enumerate(
                [
                    esc(line[3:])
                    for line in source.splitlines()
                    if line.startswith("## ")
                ],
                1,
            )
        )
        + "</div>"
    )
    return shell(
        '<article class="panel guide">' + toc + "".join(parts) + "</article>",
        "Guide",
        "guide",
    )


def experiments_page(config, dialect="sahidic", task="acceptance"):
    dialect = Dialects().require(dialect)
    tasks = {
        "acceptance": "TAB inclusion",
        "manual_order": "TAB priority",
        "crum_style": "Crum ordering",
    }
    if task not in tasks:
        raise ValueError("Unknown evaluation task")
    data = json.loads((config.reports / "ablation_results.json").read_text())
    options = "".join(
        f'<option value="{d}"'
        + (" selected" if d == dialect else "")
        + f'>{v["name"]}</option>'
        for d, v in Dialects().data.items()
    )
    taskoptions = "".join(
        f'<option value="{k}"' + (" selected" if k == task else "") + f">{v}</option>"
        for k, v in tasks.items()
    )
    body = f"""<section class="panel"><span class="eyebrow">MATCHED-CANDIDATE ABLATIONS</span><h2>Ablation Results</h2><p>Five-fold evaluation grouped by spelling family, excluding identical Coptic texts across train and validation. These tables document earlier rankers. <a href="/reports/importance_audit/REPORT.md">Current learned inclusion model, punctuation/pronoun audit, and hard-negative experiment</a>.</p>
<form class="filters" method="get"><label>Dialect <select name="dialect">{options}</select></label><label>Evaluation task <select name="task">{taskoptions}</select></label><button>Apply</button></form>
<p class="muted">Metrics compare rankings on matched candidate sets.</p></section>"""
    if dialect == "bohairic":
        body += '<p class="notice">Bohairic evaluation is provisional: no mixed-label TAB groups, one ordered TAB group, and four ordered Crum groups.</p>'
    fmt = lambda v: "N/A" if v is None else f"{v:.4f}"
    for result in data["results"]:
        if result["dialect"] != dialect:
            continue
        semantic = result["scope"] != "full_cheap"
        body += (
            '<section class="panel"><h3>'
            + ("BGE cached subset" if semantic else "Full feature set")
            + "</h3>"
        )
        body += (
            f'<p>{result["rows"]:,} rows; {result["summaries"][0][task]["groups"]} groups. '
            + (
                "Complete BGE-cache cases only; compare methods within this table."
                if semantic
                else "No semantic encoder is loaded at inference."
            )
            + "</p>"
        )
        if semantic:
            body += '<p class="formula">Original Crum4 formula = 0.55 × BGE cosine + 0.30 × ROUGE-L + 0.15 × frequency score</p>'
        body += '<div class="table-scroll"><table><thead><tr><th>Method</th><th>Features</th><th>NDCG@6 ↑</th><th>Pairwise accuracy ↑</th><th>Labeled P@3 ↑</th></tr></thead><tbody>'
        for r in result["summaries"]:
            m = r[task]
            selected = r["setup"] == "tree_full"
            body += (
                ('<tr class="chosen">' if selected else "<tr>")
                + f'<td>{esc(NAMES[r["setup"]])}'
                + (
                    ' <span class="small-tag">Default</span>'
                    if selected and dialect == "sahidic"
                    else ""
                )
                + f'</td><td>{r["features"] or "—"}</td><td>{fmt(m["ndcg_at_6"])}</td><td>{fmt(m["pair_accuracy"])}</td><td>{fmt(m.get("known_precision_at_3")) if task=="acceptance" else "N/A"}</td></tr>'
            )
        body += '</tbody></table></div><details><summary>Paired differences and 95% intervals</summary><div class="table-scroll"><table><thead><tr><th>Comparison (left minus right)</th><th>ΔNDCG</th><th>95% paired CI</th></tr></thead><tbody>'
        for c in result["comparisons"]:
            m = c["by_task"][task]
            interval = (
                "N/A"
                if m["ci95"] is None
                else f'[{m["ci95"][0]:+.4f}, {m["ci95"][1]:+.4f}]'
            )
            delta = "—" if m["delta_ndcg"] is None else f'{m["delta_ndcg"]:+.4f}'
            body += f'<tr><td>{esc(NAMES[c["setup"]])} − {esc(NAMES[c["reference"]])}</td><td>{delta}</td><td>{interval}</td></tr>'
        body += "</tbody></table></div><p>Intervals use 1,000 spelling-family cluster bootstrap samples; no multiple-comparison correction.</p></details></section>"
    diag = json.loads((config.reports / "barley_loo_diagnostic.json").read_text())
    body += '<section class="panel"><h3>barley/father negative-label diagnostic</h3><p>Four leave-one-source-out anchor tests. Cross-sense labels are inferred, not manually verified.</p><table><thead><tr><th>Method</th><th>Correct-sense ranking</th></tr></thead><tbody>'
    for name, r in diag["summaries"].items():
        body += f'<tr><td>{esc(name)}</td><td>{r["correct"]}/{r["total"]}</td></tr>'
    body += "</tbody></table><p>Both variants score 4/4; cross-sense negatives are not enabled by default.</p></section>"
    comparison = config.reports / "deployed_cohort_comparison.json"
    if comparison.exists():
        body += '<section class="panel"><h3>Full-corpus retrieval comparison</h3><p>A fixed cohort of 28 entries; model-based proxy scores are not expert judgments.</p><table><thead><tr><th>Dialect / entries</th><th>Dual-task Baseline</th><th>Joint ranker</th><th>Quality-first</th></tr></thead><tbody>'
        for r in json.loads(comparison.read_text())["summary"]:
            body += f'<tr><td>{esc(r["dialect"])} / {r["entries"]}</td><td>{r["cheap_mean_quality"]:.4f}</td><td>{r["research_mean_quality"]:.4f}</td><td>{r["quality_first_mean_quality"]:.4f}</td></tr>'
        body += "</tbody></table><p>Quality-first improves proxy scores but reduces held-out ranking performance; expert evaluation remains necessary.</p></section>"
    body += """<section class="panel"><h3>Limitations</h3><p>The default search learns dictionary-conditioned inclusion from TAB and Crum examples. A rule-gated UD ordering method remains as a separate baseline. Tables above document the earlier 45-feature experiments. The current UD-only model has a <a href="/reports/DEFINITION_FIRST.md">separate evaluation report</a>. Bohairic has limited training evidence and remains provisional.</p><p><a href="/reports/ABLATIONS.md">Ablation report</a> · <a href="/reports/ablation_results.json">Raw results</a> · <a href="/reports/LEGACY_REVIEW.md">Legacy review</a></p></section>"""
    return shell(body, "Evaluation", "experiments")
