#!/usr/bin/env python3
"""Small web UI for corpus-wide Coptic example search."""

from __future__ import annotations

import html
import json
import re
import csv
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from retrieval import (
    DEFAULT_LEXICON,
    Lexicon,
    RetrievalEngine,
    _english_tokens,
    normalize_pos,
)


PORT = 8001
MAX_BODY_BYTES = 64 * 1024
HTML_FILE = Path(__file__).with_name("index.html")
TEMPLATE_FILE = Path(__file__).with_name("templates") / "index.html"
STATIC_DIR = Path(__file__).with_name("static")
CDO_PUBLIC_DIR = Path(__file__).resolve().parents[3] / "dictionary" / "public"
CDO_DATA_DIR = Path(__file__).with_name("data")
CDO_FALLBACK_DIR = Path(__file__).resolve().parents[3] / "dictionary"
TLA_RE = re.compile(r"[A-Z]\d+")
LEXICON = Lexicon.load(DEFAULT_LEXICON)
ENGINE = RetrievalEngine()
HTML_TEMPLATE = TEMPLATE_FILE.read_text(encoding="utf-8")
CDO_EXAMPLES: dict[str, list[dict[str, str]]] = {}


def load_cdo_examples() -> dict[str, list[dict[str, str]]]:
    examples: dict[str, list[dict[str, str]]] = {}
    for filename, dialect in (
        ("examples_sahidic.tab", "Sahidic"),
        ("examples_bohairic.tab", "Bohairic"),
    ):
        path = CDO_DATA_DIR / filename
        if not path.is_file():
            path = CDO_FALLBACK_DIR / filename
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.reader(stream, delimiter="\t"):
                if len(row) < 8:
                    continue
                examples.setdefault(row[1], []).append({
                    "lemma": row[0],
                    "tla": row[1],
                    "pos": row[2],
                    "sent": row[3].replace("**", ""),
                    "translation": row[4],
                    "title": row[5],
                    "urn": row[6],
                    "priority": row[7],
                    "dialect": row[8] if len(row) > 8 else dialect,
                })
    return examples


CDO_EXAMPLES = load_cdo_examples()


def _render_results(rows: list[dict], cdo: bool = False) -> str:
    if not rows:
        return '<p class="empty">No examples for this entry in the local CDO data.</p>' if cdo else '<p class="empty">No matching examples.</p>'
    rendered = []
    for rank, row in enumerate(rows, 1):
        if cdo:
            title = html.escape(row["title"])
            citation = html.escape(row["urn"])
            metadata = f'<div class="citation">{title} · {citation} · {html.escape(row["dialect"])}</div>'
            score = f'<span class="badge">{html.escape(row["pos"])}</span>'
        else:
            metadata = (
                f'<div class="metrics">cosine {row["cosine"]:.3f} · ROUGE-L '
                f'{row["rouge_l"]:.3f} · frequency {row["frequency_score"]:.3f}</div>'
                f'<div class="citation">{html.escape(row["source"])} · {html.escape(row["ref"])}</div>'
            )
            score = f'<span class="badge">#{rank} · {row["score"]:.3f}</span>'
        rendered.append(f"""
          <article class="example-card">
            <div class="card-top">{score}<span class="badge muted">{rank}</span></div>
            <div class="coptic">{html.escape(row["sent"])}</div>
            <div class="translation">{html.escape(row.get("translation", "") or "(no translation)")}</div>
            {metadata}
          </article>""")
    return "\n".join(rendered)


def render_page(
    message: str = "",
    result: dict | None = None,
    query: str = "",
    pos: str = "",
    definition: str = "",
    compare: bool = True,
) -> str:
    content = f'<p class="error" role="alert">{html.escape(message)}</p>' if message else ""
    results_html = ""
    if result is not None:
        for warning in result.get("warnings", []):
            content += f'<p class="warning" role="status">{html.escape(warning)}</p>'
        content += (
            f'<p class="summary">Entry <b>{html.escape(result["tla"])}</b> · '
            f'<span class="coptic inline">{html.escape(result["lemma"])}</span> · '
            f'<span class="badge">{html.escape(result["pos"])}</span> · '
            f'{html.escape(result["definition"])}; {result["candidate_count"]} corpus matches, '
            f'{result["scored_count"]} scored after coarse filter.</p>'
        )
        right = _render_results(result["results"])
        if result.get("compare"):
            left = _render_results(result.get("cdo_examples", []), cdo=True)
            results_html = f"""
            <section class="comparison" aria-label="CDO and Crum4 example comparison">
              <div class="comparison-panel"><h2>CDO examples</h2>{left}</div>
              <div class="comparison-panel"><h2>Crum4 corpus search</h2>{right}</div>
            </section>"""
        else:
            results_html = f'<section class="single-results"><h2>Crum4 corpus search</h2>{right}</section>'
    return (
        HTML_TEMPLATE.replace("{{QUERY}}", html.escape(query))
        .replace("{{POS}}", html.escape(pos))
        .replace("{{DEFINITION}}", html.escape(definition))
        .replace("{{COMPARE_CHECKED}}", "checked" if compare else "")
        .replace("{{CONTENT}}", content)
        .replace("{{RESULTS}}", results_html)
    )


def _filter_cdo_examples(rows: list[dict], gloss: str) -> list[dict]:
    """Keep only CDO examples whose translation overlaps with the entry gloss.

    C998 (barley) and C992 (father) share lemma + NOUN; bohairic father
    examples would otherwise leak into C998's left panel.
    """
    if not rows:
        return rows
    gloss_tokens = set(_english_tokens(gloss or ""))
    if gloss_tokens:
        matched = [
            r for r in rows
            if set(_english_tokens(r.get("translation", ""))) & gloss_tokens
        ]
        if matched:
            return matched
    return rows


def lookup(fields: dict[str, list[str]]) -> dict:
    query = fields.get("query", [""])[0].strip()
    pos_field = fields.get("pos", [""])[0].strip()
    definition_override = fields.get("definition", [""])[0].strip()
    try:
        if TLA_RE.fullmatch(query.upper()):
            entry = LEXICON.resolve_tla(query)
        else:
            if not pos_field:
                raise ValueError("Enter a TLA ID or provide both a lemma and POS.")
            entry = LEXICON.resolve_lemma(query, pos_field, definition_override)
        if pos_field and normalize_pos(pos_field) != entry.pos:
            raise ValueError(
                f"POS mismatch: this entry is {entry.pos}, not {normalize_pos(pos_field) or pos_field}."
            )
        definition = definition_override or entry.gloss
        result = ENGINE.search(
            entry.lemma,
            entry.pos or "",
            definition,
            top_n=int(fields.get("top_n", ["10"])[0]),
            candidate_limit=int(fields.get("candidate_limit", ["100"])[0]),
        )
        result["tla"] = entry.tla or query
        result["compare"] = fields.get("compare", [""])[0].casefold() in {"1", "true", "on", "yes"}
        if result["compare"]:
            result["cdo_examples"] = _filter_cdo_examples(
            CDO_EXAMPLES.get(entry.tla or query, []),
            entry.gloss,
        )
        return result
    except (ValueError, LookupError, FileNotFoundError, RuntimeError) as error:
        raise ValueError(str(error)) from error


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.startswith("/static/"):
            self._send_static(self.path.removeprefix("/static/"))
            return
        if self.path not in {"/", "/index.html"}:
            self._send(404, "text/html; charset=utf-8", render_page("Not found."))
            return
        self._send(200, "text/html; charset=utf-8", render_page())

    def do_POST(self) -> None:
        if self.path not in {"/search", "/api/search"}:
            self._send(404, "application/json", json.dumps({"error": "Not found."}))
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > MAX_BODY_BYTES:
                raise ValueError(f"Request body must be at most {MAX_BODY_BYTES} bytes.")
            body = self.rfile.read(length).decode("utf-8", errors="replace")
            if self.path == "/api/search":
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise ValueError("JSON request body must be an object.")
                fields = {
                    str(key): [str(value)]
                    for key, value in payload.items()
                }
            else:
                fields = parse_qs(body, keep_blank_values=True)
            result = lookup(fields)
        except (ValueError, LookupError, FileNotFoundError, RuntimeError, json.JSONDecodeError) as error:
            if self.path == "/api/search":
                self._send(400, "application/json; charset=utf-8", json.dumps({"error": str(error)}, ensure_ascii=False))
            else:
                self._send(
                    400,
                    "text/html; charset=utf-8",
                    render_page(
                        str(error),
                        query=fields.get("query", [""])[0],
                        pos=fields.get("pos", [""])[0],
                        definition=fields.get("definition", [""])[0],
                        compare=fields.get("compare", [""])[0] == "on",
                    ),
                )
            return
        if self.path == "/api/search":
            self._send(
                200,
                "application/json; charset=utf-8",
                json.dumps(result, ensure_ascii=False),
            )
        else:
            self._send(
                200,
                "text/html; charset=utf-8",
                render_page(
                    result=result,
                    query=fields.get("query", [""])[0],
                    pos=fields.get("pos", [""])[0],
                    definition=fields.get("definition", [""])[0],
                    compare=result.get("compare", False),
                ),
            )

    def _send_static(self, relative_path: str) -> None:
        static_files = {
            "style.css": (STATIC_DIR / "style.css", "text/css; charset=utf-8"),
            "antinoou-webfont.woff": (CDO_PUBLIC_DIR / "antinoou-webfont.woff", "font/woff"),
            "antinoou-webfont.ttf": (CDO_PUBLIC_DIR / "antinoou-webfont.ttf", "font/ttf"),
        }
        target = static_files.get(relative_path)
        if target is None or not target[0].is_file():
            self._send(404, "text/plain; charset=utf-8", "Not found.")
            return
        data = target[0].read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", target[1])
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send(self, status: int, content_type: str, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format_string: str, *args: object) -> None:
        print(f"[HTTP] {self.address_string()} {format_string % args}")


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Crum4 full-corpus search at http://localhost:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[STOP]")
    finally:
        server.server_close()
