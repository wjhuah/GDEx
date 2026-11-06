import argparse
import json
from pathlib import Path
from .config import Config, PROJECT, write_json
from .dialects import Dialects


def main():
    parser = argparse.ArgumentParser(
        description="Coptic dictionary examples, with auditable sense and quality evidence"
    )
    parser.add_argument("--source-root", type=Path, default=Config().source)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Index every TT example; no candidate cap")
    build.add_argument("--force", action="store_true")
    sub.add_parser(
        "study", help="First experiment: expert C992/C998 cross-sense contrast"
    )
    sub.add_parser(
        "dataset",
        help="Rebuild TAB/Crum supervision and all dictionary candidate links",
    )
    sub.add_parser(
        "features", help="Compute fixed GDEX/native dependency and real BGE features"
    )
    train = sub.add_parser(
        "train",
        help="Train quality and Crum-style models separately for every registered dialect",
    )
    search = sub.add_parser(
        "search", help="Search the full corpus for a dictionary TLA or unique lemma"
    )
    search.add_argument("query")
    search.add_argument("--pos")
    search.add_argument("--definition")
    search.add_argument("--top", type=int, default=10)
    search.add_argument(
        "--ranker",
        choices=[
            "cheap",
            "learned",
            "research",
            "unified",
            "quality_first",
            "linear",
            "definition_syntax",
        ],
        default="research",
    )
    search.add_argument("--dialect", choices=list(Dialects().data), default="sahidic")
    search.add_argument(
        "--semantic-limit",
        type=int,
        default=0,
        help="0 scores all retrieved candidates; positive values are an explicit optional cap",
    )
    search.add_argument("--output", type=Path)
    shortlist = sub.add_parser(
        "shortlist",
        help="Cheap ML plus syntax diversity: around ten candidates, no LLM",
    )
    shortlist.add_argument("tla")
    shortlist.add_argument(
        "--dialect", choices=list(Dialects().data), default="sahidic"
    )
    shortlist.add_argument("--size", type=int, default=10)
    server = sub.add_parser("serve", help="Open local expert/corpus comparison UI")
    server.add_argument("--port", type=int, default=8012)
    args = parser.parse_args()
    cfg = Config(source=args.source_root.resolve())
    if args.command == "build":
        from .corpus import Corpus

        result = Corpus(cfg).build(args.force)
        write_json(cfg.reports / "corpus_inventory.json", result)
    elif args.command == "study":
        from .study import run

        result = run(cfg)
    elif args.command == "dataset":
        from .dataset import build

        result = build(cfg)
    elif args.command == "features":
        from .training import feature_data

        result = feature_data(cfg)
    elif args.command == "train":
        from .shortlist import train

        result = train(cfg)
    elif args.command == "search":
        from .retrieval import Search

        result = Search(cfg).search(
            args.query,
            args.top,
            args.ranker,
            args.semantic_limit,
            args.pos,
            args.definition,
            args.dialect,
        )
        if args.output:
            write_json(args.output, result)
    elif args.command == "serve":
        from .web import serve

        serve(cfg, args.port)
        return
    elif args.command == "shortlist":
        from .shortlist import Shortlist, export

        result = export(cfg, Shortlist(cfg, args.dialect).search(args.tla, args.size))
    print(json.dumps(result, ensure_ascii=False, indent=2))
