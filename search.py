#!/usr/bin/env python3
"""Command-line interface for Crum4 full-corpus example retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from retrieval import DEFAULT_LEXICON, Lexicon, RetrievalEngine, normalize_pos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--tla", help="KELLIA TLA ID, e.g. C998")
    target.add_argument("--lemma", help="Coptic lemma; requires --pos")
    parser.add_argument("--pos", help="POS (UD tag or supported KELLIA label)")
    parser.add_argument("--gloss", default="", help="Optional English definition/gloss")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    args = parser.parse_args()

    try:
        lexicon = Lexicon.load(args.lexicon)
        if args.tla:
            entry = lexicon.resolve_tla(args.tla)
            if args.pos and normalize_pos(args.pos) != entry.pos:
                parser.error(
                    f"{args.tla} is tagged {entry.pos}, not "
                    f"{normalize_pos(args.pos) or args.pos}."
                )
        else:
            if not args.pos:
                parser.error("--lemma requires --pos.")
            entry = lexicon.resolve_lemma(args.lemma, args.pos, args.gloss)
        result = RetrievalEngine().search(
            entry.lemma,
            entry.pos or "",
            args.gloss or entry.gloss,
            top_n=args.top_n,
            candidate_limit=args.candidate_limit,
        )
    except (LookupError, ValueError, FileNotFoundError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
