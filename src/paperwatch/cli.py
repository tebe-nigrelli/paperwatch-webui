from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from importlib.resources import as_file, files
from pathlib import Path
from typing import Sequence

from .config import AppConfig, ConfigError, load_config
from .blog_import import import_blog_links_path
from .blogs import (
    SOURCE_MODES,
    normalize_blog_url,
    validate_article_url_pattern,
    validate_content_selector,
)
from .db import Database
from .default_queries import add_default_queries, refresh_default_regexp_patterns
from .llm import LLMClient, create_llm
from .pipeline import PaperWatch
from .sources.arxiv import ArxivClient
from .sources.openalex import OpenAlexClient


@dataclass
class Services:
    config: AppConfig
    db: Database
    openalex: OpenAlexClient
    arxiv: ArxivClient | None
    llm: LLMClient

    def close(self) -> None:
        self.openalex.close()
        if self.arxiv:
            self.arxiv.close()
        self.llm.close()
        self.db.close()


def build_services(config: AppConfig) -> Services:
    db = Database(config.paths.database)
    if db.was_created:
        model_profile = config.available_llms[0].profile if config.available_llms else ""
        add_default_queries(db, model_profile)
    refresh_default_regexp_patterns(db)
    return Services(
        config=config,
        db=db,
        openalex=OpenAlexClient(
            api_key=config.openalex.api_key,
            base_url=config.openalex.base_url,
            timeout_seconds=config.openalex.timeout_seconds,
        ),
        arxiv=(
            ArxivClient(
                base_url=config.arxiv.base_url,
                timeout_seconds=config.arxiv.timeout_seconds,
                min_interval_seconds=config.arxiv.min_interval_seconds,
                title_match_threshold=config.arxiv.title_match_threshold,
            )
            if config.arxiv.enabled
            else None
        ),
        llm=create_llm(config.llm),
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an ISO date in YYYY-MM-DD format") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paperwatch")
    parser.add_argument("--config", default="config.toml", help="Path to TOML configuration")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create a starter configuration")
    init.add_argument("path", nargs="?", default="config.toml")

    config_cmd = sub.add_parser("config", help="Edit primary LLM configuration")
    config_sub = config_cmd.add_subparsers(dest="config_command", required=True)
    provider_cmd = config_sub.add_parser(
        "provider", help="Set [llm].provider while preserving the rest of config.toml"
    )
    provider_cmd.add_argument(
        "provider", choices=("ollama", "openai", "codex", "claude-code", "none")
    )

    author = sub.add_parser("author", help="Manage tracked authors")
    author_sub = author.add_subparsers(dest="author_command", required=True)
    search = author_sub.add_parser("search")
    search.add_argument("name")
    search.add_argument("--limit", type=_positive_int, default=10)
    add = author_sub.add_parser("add")
    add.add_argument("openalex_id")
    author_sub.add_parser("list")
    enable = author_sub.add_parser("enable")
    enable.add_argument("openalex_id")
    disable = author_sub.add_parser("disable")
    disable.add_argument("openalex_id")
    remove = author_sub.add_parser("remove")
    remove.add_argument("openalex_id")

    blog = sub.add_parser("blog", help="Manage author blog links")
    blog_sub = blog.add_subparsers(dest="blog_command", required=True)
    blog_add = blog_sub.add_parser("add", help="Link a blog to an author")
    blog_add.add_argument("author_id")
    blog_add.add_argument("url")
    blog_add.add_argument("--label", default="")
    blog_add.add_argument("--relationship", default="")
    blog_add.add_argument("--mode", choices=sorted(SOURCE_MODES), default="auto")
    blog_add.add_argument("--selector", default="")
    blog_add.add_argument("--article-url-pattern", default="")
    blog_import = blog_sub.add_parser("import", help="Import author blog links from CSV")
    blog_import.add_argument("csv_path")
    blog_import.add_argument(
        "--create-missing",
        action="store_true",
        help="Create paused, blog-only authors when no exact name match exists",
    )
    blog_sub.add_parser("list", help="List stored blog links")

    run = sub.add_parser("run", help="Fetch recent papers and author blogs")
    run.add_argument("--today", type=_iso_date, help="Testing override in YYYY-MM-DD format")

    backfill = sub.add_parser(
        "backfill",
        help="Fetch a historical window, download open PDFs, and skip automatic analysis",
    )
    backfill.add_argument(
        "--days",
        type=_positive_int,
        default=365,
        help="Number of calendar days to fetch (default: 365)",
    )
    backfill.add_argument(
        "--max-papers-per-author",
        type=_positive_int,
        default=5000,
        help="Safety cap per author for the backfill (default: 5000)",
    )
    backfill.add_argument(
        "--today", type=_iso_date, help="Testing override in YYYY-MM-DD format"
    )

    papers = sub.add_parser("papers", help="List stored papers")
    papers.add_argument("--limit", type=_positive_int, default=20)

    runs = sub.add_parser("runs", help="List previous runs")
    runs.add_argument("--limit", type=_positive_int, default=20)

    web = sub.add_parser("web", help="Run the local web interface")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=_positive_int, default=8765)
    web.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default="info",
    )
    return parser


def _init_config(path: str) -> int:
    destination = Path(path).expanduser()
    if destination.exists():
        print(f"Refusing to overwrite {destination}", file=sys.stderr)
        return 2
    destination.parent.mkdir(parents=True, exist_ok=True)
    template_ref = files("paperwatch").joinpath("defaults/config.toml")
    prompt_ref = files("paperwatch").joinpath("defaults/analyze.txt")
    with as_file(template_ref) as template:
        shutil.copyfile(template, destination)
    prompt_destination = destination.parent / "prompts" / "analyze.txt"
    prompt_destination.parent.mkdir(parents=True, exist_ok=True)
    if not prompt_destination.exists():
        with as_file(prompt_ref) as prompt_source:
            shutil.copyfile(prompt_source, prompt_destination)
    print(destination)
    return 0


def _config_command(args: argparse.Namespace) -> int:
    if args.config_command != "provider":
        return 2
    path = Path(args.config).expanduser()
    if not path.is_file():
        print(f"Config file not found: {path}", file=sys.stderr)
        return 2
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    section_match = re.search(r"(?ms)^\[llm\][ \t]*\n(?P<body>.*?)(?=^\[|\Z)", text)
    if section_match is None:
        print("config.toml has no [llm] section", file=sys.stderr)
        return 2
    body = section_match.group("body")
    provider_line = f'provider = "{args.provider}"'
    if re.search(r"(?m)^[ \t]*provider[ \t]*=", body):
        body = re.sub(
            r'(?m)^[ \t]*provider[ \t]*=[ \t]*[^\n#]*(?P<comment>[ \t]*#.*)?$',
            lambda match: provider_line + (match.group("comment") or ""),
            body,
            count=1,
        )
    else:
        body = provider_line + "\n" + body
    if args.provider in {"codex", "claude-code"}:
        command_name = "codex" if args.provider == "codex" else "claude"
        command_line = f'command = "{command_name}"'
        if re.search(r"(?m)^[ \t]*command[ \t]*=", body):
            body = re.sub(
                r'(?m)^[ \t]*command[ \t]*=[ \t]*[^\n#]*(?P<comment>[ \t]*#.*)?$',
                lambda match: command_line + (match.group("comment") or ""),
                body,
                count=1,
            )
        else:
            provider_match = re.search(r"(?m)^provider[^\n]*\n", body)
            offset = provider_match.end() if provider_match else 0
            body = body[:offset] + command_line + "\n" + body[offset:]
    updated = text[: section_match.start("body")] + body + text[section_match.end("body") :]
    try:
        path.write_text(updated, encoding="utf-8")
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.provider == "codex":
        print(f'{path}: [llm] provider = "codex"; using installed codex CLI and its existing login')
    else:
        print(f'{path}: [llm] provider = "{args.provider}"')
    return 0


def _author_command(args: argparse.Namespace, services: Services) -> int:
    if args.author_command == "search":
        results = services.openalex.search_authors(args.name, limit=args.limit)
        for index, author in enumerate(results, 1):
            institution = author.institution or "unknown institution"
            orcid = author.orcid or "no ORCID"
            print(
                f"{index}. {author.name} | {institution} | {author.openalex_id} | "
                f"{orcid} | works={author.works_count} citations={author.cited_by_count}"
            )
        return 0
    if args.author_command == "add":
        author = services.openalex.get_author(args.openalex_id)
        services.db.add_author(author)
        print(f"Added {author.name} ({author.openalex_id})")
        return 0
    if args.author_command == "list":
        for author in services.db.list_authors(tracked_only=True):
            state = "enabled" if author["enabled"] else "disabled"
            institution = author["institution"] or "unknown institution"
            print(f"{author['openalex_id']} | {author['name']} | {institution} | {state}")
        return 0
    if args.author_command in {"enable", "disable"}:
        enabled = args.author_command == "enable"
        changed = services.db.set_author_enabled(args.openalex_id, enabled)
        if not changed:
            print("Author not found", file=sys.stderr)
            return 1
        print(f"{args.openalex_id}: {'enabled' if enabled else 'disabled'}")
        return 0
    if args.author_command == "remove":
        if not services.db.remove_author(args.openalex_id):
            print("Author not found", file=sys.stderr)
            return 1
        print(f"Removed {args.openalex_id} from tracking; database record retained")
        return 0
    return 2


def _run(args: argparse.Namespace, services: Services) -> int:
    app = PaperWatch(
        services.config,
        db=services.db,
        openalex=services.openalex,
        arxiv=services.arxiv,
        llm=services.llm,
    )
    try:
        stats = app.run(today=args.today)
    finally:
        app.close()
    print(
        f"run={stats.run_id} status={stats.status} authors={stats.authors_checked} "
        f"papers={stats.papers_found} analyses={stats.papers_analyzed} "
        f"pdfs={stats.pdfs_downloaded} blogs={getattr(stats, 'blogs_checked', 0)} "
        f"blog_updates={getattr(stats, 'blog_updates_found', 0)} errors={stats.errors_count}"
    )
    return 0 if stats.status == "success" else 1


def _backfill(args: argparse.Namespace, services: Services) -> int:
    current_day = args.today or datetime.now(timezone.utc).date()
    since = current_day - timedelta(days=args.days)
    last_counts: tuple[int, int, int, int] | None = None

    def progress(kind: str, value: str) -> None:
        nonlocal last_counts
        if kind == "stage":
            print(value, flush=True)
            return
        if kind == "message":
            print(value, flush=True)
            return
        if kind != "run_stats":
            return
        values = json.loads(value)
        counts = (
            int(values.get("authors_checked", 0)),
            int(values.get("papers_found", 0)),
            int(values.get("pdfs_downloaded", 0)),
            int(values.get("errors_count", 0)),
        )
        if counts == last_counts:
            return
        last_counts = counts
        print(
            f"authors={counts[0]} papers={counts[1]} pdfs={counts[2]} errors={counts[3]}",
            flush=True,
        )

    app = PaperWatch(
        services.config,
        db=services.db,
        openalex=services.openalex,
        arxiv=services.arxiv,
        llm=services.llm,
        progress=progress,
    )
    try:
        stats = app.run(
            today=current_day,
            since_override=since,
            download_only=True,
            historical=True,
            max_papers_per_author_override=args.max_papers_per_author,
        )
    finally:
        app.close()
    print(
        f"backfill={stats.run_id} status={stats.status} since={since.isoformat()} "
        f"authors={stats.authors_checked} papers={stats.papers_found} "
        f"pdfs={stats.pdfs_downloaded} analyses={stats.papers_analyzed} "
        f"errors={stats.errors_count}"
    )
    return 0 if stats.status == "success" else 1


def _list_papers(args: argparse.Namespace, services: Services) -> int:
    for paper in services.db.list_papers(args.limit):
        published = paper["publication_date"] or "unknown date"
        local = paper["local_pdf_path"] or "no local PDF"
        print(f"{paper['id']} | {published} | {paper['title']} | {local}")
    return 0


def _list_runs(args: argparse.Namespace, services: Services) -> int:
    for run in services.db.list_runs(args.limit):
        print(
            f"{run['id']} | {run['started_at']} | {run['status']} | "
            f"authors={run['authors_checked']} papers={run['papers_found']} "
            f"analyses={run['papers_analyzed']} errors={run['errors_count']}"
        )
    return 0


def _blog_command(args: argparse.Namespace, services: Services) -> int:
    if args.blog_command == "add":
        url = normalize_blog_url(args.url)
        services.db.add_author_blog(
            args.author_id,
            url,
            label=args.label,
            relationship=args.relationship,
            source_mode=args.mode,
            content_selector=validate_content_selector(args.selector),
            article_url_pattern=validate_article_url_pattern(
                args.article_url_pattern
            ),
        )
        print(f"Linked {url} to {args.author_id}")
        return 0
    if args.blog_command == "import":
        stats = import_blog_links_path(
            services.db,
            args.csv_path,
            create_missing=args.create_missing,
        )
        print(
            f"rows={stats.rows} links={stats.links} created={stats.authors_created} "
            f"missing={stats.authors_missing} ambiguous={stats.authors_ambiguous} "
            f"invalid_urls={stats.invalid_urls}"
        )
        return 0 if not stats.authors_ambiguous else 1
    if args.blog_command == "list":
        for author in services.db.list_authors(tracked_only=True):
            for blog in services.db.author_blogs(str(author["openalex_id"])):
                print(
                    f"{author['openalex_id']} | {author['name']} | "
                    f"{blog['label'] or blog['title'] or 'Blog'} | "
                    f"{blog['source_mode']} | {blog['url']}"
                )
        return 0
    return 2



def _serve_web(args: argparse.Namespace) -> int:
    import uvicorn

    from .web import create_app

    while True:
        app = create_app(args.config)
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
        state = getattr(app, "state", None)
        if not bool(getattr(state, "restart_requested", False)):
            return 0

def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init":
        return _init_config(args.path)
    if args.command == "config":
        return _config_command(args)
    if args.command == "web":
        try:
            return _serve_web(args)
        except (ConfigError, OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2

    try:
        config = load_config(args.config)
        services = build_services(config)
    except (ConfigError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        if args.command == "author":
            return _author_command(args, services)
        if args.command == "blog":
            return _blog_command(args, services)
        if args.command == "run":
            return _run(args, services)
        if args.command == "backfill":
            return _backfill(args, services)
        if args.command == "papers":
            return _list_papers(args, services)
        if args.command == "runs":
            return _list_runs(args, services)
        return 2
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        services.close()


if __name__ == "__main__":
    raise SystemExit(main())
