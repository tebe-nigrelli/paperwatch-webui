from __future__ import annotations

import csv
import hashlib
import html
import inspect
import io
import os
import sqlite3
import tempfile
import zipfile
import json
import re
import shutil
import subprocess
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import quote_plus, urlencode, urlparse

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTask
from starlette.datastructures import UploadFile

from . import __version__
from .analysis_jobs import AnalysisController
from .blog_import import import_blog_links
from .blogs import (
    SOURCE_MODES,
    BlogFetch,
    blog_diff,
    compact_blog_diff,
    fetch_blog,
    infer_source_mode,
    normalize_blog_url,
    normalize_source_mode,
    validate_article_url_pattern,
    validate_content_selector,
)
from .config import AppConfig, load_config
from .db import Database
from .default_queries import add_default_queries
from .job_runtime import ProgressCallback
from .locking import AnalysisAlreadyInProgress, analysis_lock
from .models import (
    AuthorCandidate,
    PaperAuthor,
    PaperRecord,
    clean_text,
    extract_arxiv_id,
    normalize_title,
    short_openalex_id,
    strip_doi,
)
from .pdfs import PDFError, download_pdf, pdf_sha256, safe_pdf_name, validate_pdf_file
from .query_jobs import QueryController
from .sources.arxiv import ArxivClient, ArxivEntry
from .sources.openalex import OpenAlexClient

PAPER_KINDS = (
    "mathematics",
    "machine_learning",
    "theoretical_machine_learning",
    "survey",
    "other",
)
SCOPES = ("abstract", "fulltext")
SORTS = ("newest", "oldest", "discovered", "title")
PAGE_SIZES = (25, 100, 250, 500, 1000)
DEFAULT_PAGE_SIZE = 25
SELECTED_CONTENT_PREVIEW_LIMIT = 25


@dataclass(frozen=True, slots=True)
class PaperFilters:
    query: str = ""
    title_query: str = ""
    author_id: str = ""
    author_ids: tuple[str, ...] = ()
    author_tags: tuple[str, ...] = ()
    content_tags: tuple[str, ...] = ()
    topic_id: str = ""
    topic_ids: tuple[str, ...] = ()
    paper_kind: str = ""
    scope: str = ""
    has_pdf: bool | None = None
    analyzed: bool | None = None
    starred: bool | None = None
    read: bool | None = None
    notes: bool | None = None
    display_stubs: bool = False
    date_from: str = ""
    date_to: str = ""
    sort: str = "newest"
    page: int = 1
    page_size: int = DEFAULT_PAGE_SIZE

    def selection(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "title_query": self.title_query,
            "author_id": self.author_id,
            "author_ids": list(self.author_ids),
            "author_tags": list(self.author_tags),
            "content_tags": list(self.content_tags),
            "topic_id": self.topic_id,
            "topic_ids": list(self.topic_ids),
            "paper_kind": self.paper_kind,
            "scope": self.scope,
            "has_pdf": self.has_pdf,
            "analyzed": self.analyzed,
            "starred": self.starred,
            "read": self.read,
            "notes": self.notes,
            "display_stubs": self.display_stubs,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "sort": self.sort,
        }

    def selection_items(self) -> list[tuple[str, str]]:
        items = [
            ("q", self.query),
            ("title", self.title_query),
            *(("author", value) for value in (self.author_ids or ((self.author_id,) if self.author_id else ()))),
            *(("author_tag", tag) for tag in self.author_tags),
            *(("tag", tag) for tag in self.content_tags),
            *(("topic", value) for value in (self.topic_ids or ((self.topic_id,) if self.topic_id else ()))),
            ("kind", self.paper_kind),
            ("scope", self.scope),
            ("has_pdf", _bool_query(self.has_pdf)),
            ("analyzed", _bool_query(self.analyzed)),
            ("starred", _bool_query(self.starred)),
            ("read", _bool_query(self.read)),
            ("notes", _bool_query(self.notes)),
            ("display_stubs", "yes" if self.display_stubs else ""),
            ("date_from", self.date_from),
            ("date_to", self.date_to),
            ("sort", self.sort if self.sort != "newest" else ""),
        ]
        return [(key, value) for key, value in items if value]

    def query_items(self, *, page: int | None = None) -> list[tuple[str, str]]:
        items = [
            ("q", self.query),
            ("title", self.title_query),
            *(("author", value) for value in (self.author_ids or ((self.author_id,) if self.author_id else ()))),
            *(("author_tag", tag) for tag in self.author_tags),
            *(("tag", tag) for tag in self.content_tags),
            *(("topic", value) for value in (self.topic_ids or ((self.topic_id,) if self.topic_id else ()))),
            ("kind", self.paper_kind),
            ("scope", self.scope),
            ("has_pdf", _bool_query(self.has_pdf)),
            ("analyzed", _bool_query(self.analyzed)),
            ("starred", _bool_query(self.starred)),
            ("read", _bool_query(self.read)),
            ("notes", _bool_query(self.notes)),
            ("display_stubs", "yes" if self.display_stubs else ""),
            ("date_from", self.date_from),
            ("date_to", self.date_to),
            ("sort", self.sort if self.sort != "newest" else ""),
            ("page_size", str(self.page_size) if self.page_size != DEFAULT_PAGE_SIZE else ""),
            ("page", str(page if page is not None else self.page)),
        ]
        return [(key, value) for key, value in items if value and value != "1"]

    @property
    def query_string(self) -> str:
        return urlencode(self.query_items())


@dataclass(slots=True)
class RunSnapshot:
    running: bool = False
    mode: str | None = None
    query: str | None = None
    paper_id: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    run_id: int | None = None
    status: str | None = None
    stats: dict[str, Any] | None = None
    error: str | None = None
    cancelled: bool = False
    cancel_requested: bool = False
    progress_current: int = 0
    progress_total: int = 0
    stage: str = ""
    message: str = ""


class RunController:
    """Allow one in-process fetch at a time without sharing SQLite connections."""

    def __init__(
        self,
        config_path: str | Path,
        runner: Callable[[Path, str], dict[str, Any]] | None = None,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self._runner = runner or _run_pipeline
        self._lock = threading.Lock()
        self._snapshot = RunSnapshot()
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()

    def start(self, mode: str = "run", query: str | None = None) -> bool:
        if mode not in {"run", "backfill", "search", "download"}:
            raise ValueError("Unknown run mode")
        normalized_query = query.strip() if query else None
        if mode in {"search", "download"} and not normalized_query:
            raise ValueError("Paper identifier cannot be empty")
        with self._lock:
            if self._snapshot.running:
                return False
            self._cancel_event = threading.Event()
            self._snapshot = RunSnapshot(
                running=True,
                mode=mode,
                query=normalized_query,
                started_at=_utcnow(),
                status="running",
                stage="Queued",
                message="Preparing fetch",
            )
            self._thread = threading.Thread(
                target=self._work,
                args=(mode, normalized_query),
                name="paperwatch-web-run",
                daemon=True,
            )
            self._thread.start()
            return True

    def _work(self, mode: str, query: str | None) -> None:
        try:
            parameter_count = len(inspect.signature(self._runner).parameters)
            if parameter_count >= 5:
                result = self._runner(
                    self.config_path,
                    mode,
                    query,
                    self._progress,
                    self._cancel_event,
                )
            elif parameter_count >= 3:
                result = self._runner(self.config_path, mode, query)
            else:
                result = self._runner(self.config_path, mode)
            with self._lock:
                self._snapshot.running = False
                self._snapshot.finished_at = _utcnow()
                self._snapshot.run_id = _int_or_none(result.get("run_id"))
                self._snapshot.paper_id = _int_or_none(result.get("paper_id"))
                self._snapshot.status = str(result.get("status") or "success")
                self._snapshot.stats = dict(result)
                self._snapshot.cancelled = self._snapshot.status == "cancelled"
                self._snapshot.cancel_requested = False
                self._snapshot.stage = (
                    "Cancelled" if self._snapshot.cancelled else "Complete"
                )
        except Exception as exc:
            with self._lock:
                self._snapshot.running = False
                self._snapshot.finished_at = _utcnow()
                self._snapshot.status = "failed"
                self._snapshot.error = str(exc)
                self._snapshot.cancel_requested = False
                self._snapshot.stage = "Failed"
                self._snapshot.message = str(exc)

    def _progress(self, kind: str, value: str) -> None:
        with self._lock:
            if kind == "stage":
                self._snapshot.stage = value[:200]
            elif kind == "message":
                self._snapshot.message = value[:1000]
            elif kind == "progress":
                current, separator, total = value.partition("/")
                if separator:
                    self._snapshot.progress_current = _int_or_none(current) or 0
                    self._snapshot.progress_total = _int_or_none(total) or 0
            elif kind == "run_stats":
                try:
                    payload = json.loads(value)
                except (TypeError, json.JSONDecodeError):
                    return
                if not isinstance(payload, dict):
                    return
                run_id = _int_or_none(payload.pop("run_id", None))
                if run_id is not None:
                    self._snapshot.run_id = run_id
                self._snapshot.stats = {
                    str(key): int(item)
                    for key, item in payload.items()
                    if isinstance(item, int)
                }

    def cancel(self) -> bool:
        with self._lock:
            if not self._snapshot.running:
                return False
            self._snapshot.cancel_requested = True
            self._snapshot.message = "Stopping after the current network operation"
            self._cancel_event.set()
        return True

    def snapshot(self) -> RunSnapshot:
        with self._lock:
            return RunSnapshot(**asdict(self._snapshot))

    def wait(self, timeout: float = 10) -> None:
        thread = self._thread
        if thread:
            thread.join(timeout)


def _run_pipeline(
    config_path: Path,
    mode: str = "run",
    query: str | None = None,
    progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    from .cli import build_services
    from .pipeline import PaperWatch

    config = load_config(config_path)
    services = build_services(config)
    app: PaperWatch | None = None
    try:
        app = PaperWatch(
            config,
            db=services.db,
            openalex=services.openalex,
            arxiv=services.arxiv,
            llm=services.llm,
            progress=progress,
            cancel_event=cancel_event,
        )
        paper_id: int | None = None
        if mode == "backfill":
            current_day = datetime.now(timezone.utc).date()
            stats = app.run(
                today=current_day,
                since_override=current_day - timedelta(days=365),
                download_only=True,
                historical=True,
                max_papers_per_author_override=5000,
            )
        elif mode == "search":
            stats, paper_id = app.fetch_title(query or "")
        elif mode == "download":
            stats, paper_id = app.fetch_work(query or "")
        elif mode == "run":
            stats = app.run()
        else:
            raise ValueError("Unknown run mode")
        return {
            "run_id": stats.run_id,
            "status": stats.status,
            "paper_id": paper_id,
            **stats.as_dict(),
        }
    finally:
        if app is not None:
            app.close()
        services.close()


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _bool_query(value: bool | None) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return ""


def _parse_tristate(value: str | None) -> bool | None:
    if value in {"yes", "true", "1"}:
        return True
    if value in {"no", "false", "0"}:
        return False
    return None


def _valid_date(value: str | None) -> str:
    if not value:
        return ""
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        return ""


def _manual_author_names(value: str) -> list[str]:
    separator = r"[\n;]+" if re.search(r"[\n;]", value) else r","
    names = [clean_text(item) for item in re.split(separator, value)]
    unique: list[str] = []
    seen: set[str] = set()
    for name in names:
        if not name:
            continue
        key = name.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(name)
    if not unique:
        raise ValueError("At least one author is required")
    if len(unique) > 200:
        raise ValueError("At most 200 authors can be imported")
    return unique


def _manual_source_url(value: str) -> str:
    source = clean_text(value)
    if not source:
        raise ValueError("Source link is required")
    if len(source) > 2000:
        raise ValueError("Source link is too long")
    parsed = urlparse(source)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Source link must be a valid http or https URL")
    return source


def _manual_openalex_author_id(value: str) -> str | None:
    cleaned = short_openalex_id(clean_text(value))
    if not cleaned:
        return None
    if not re.fullmatch(r"A\d+", cleaned):
        raise ValueError("OpenAlex author ID must look like A123456789")
    return cleaned


def _manual_publication_date(value: str) -> str | None:
    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        return date.fromisoformat(cleaned).isoformat()
    except ValueError as exc:
        raise ValueError("Publication date must use YYYY-MM-DD") from exc


def _manual_paper_record(values: dict[str, str], *, filename: str) -> PaperRecord:
    title = clean_text(values.get("title"))
    if not title:
        raise ValueError("Title is required")
    if len(title) > 1000:
        raise ValueError("Title is too long")
    authors = _manual_author_names(values.get("authors", ""))
    source_url = _manual_source_url(values.get("source_link", ""))
    abstract = clean_text(values.get("abstract"))
    if abstract and len(abstract) > 1_000_000:
        raise ValueError("Abstract is too long")
    venue = clean_text(values.get("venue"))
    if venue and len(venue) > 500:
        raise ValueError("Venue is too long")
    doi = strip_doi(values.get("doi"))
    arxiv_raw = clean_text(values.get("arxiv_id"))
    arxiv_id = extract_arxiv_id(f"arxiv:{arxiv_raw}") if arxiv_raw else None
    if arxiv_raw and not arxiv_id:
        raise ValueError("arXiv ID is invalid")
    publication_date = _manual_publication_date(values.get("publication_date", ""))
    paper_authors = [
        PaperAuthor(
            openalex_id=None,
            name=name,
            position=(
                "first"
                if index == 0
                else "last"
                if index == len(authors) - 1
                else "middle"
            ),
        )
        for index, name in enumerate(authors)
    ]
    return PaperRecord(
        openalex_id=None,
        doi=doi,
        arxiv_id=arxiv_id,
        title=title,
        abstract=abstract,
        publication_date=publication_date,
        updated_date=None,
        venue=venue,
        publication_url=source_url,
        pdf_url=None,
        abstract_source="manual" if abstract else None,
        pdf_source="manual_upload",
        authors=paper_authors,
        metadata={
            "manual_import": True,
            "original_filename": Path(filename).name,
            "source_link": source_url,
            "imported_at": _utcnow(),
        },
    )


def _paper_import_values(paper: PaperRecord) -> dict[str, str]:
    return {
        "title": paper.title,
        "authors": "\n".join(author.name for author in paper.authors),
        "source_link": paper.publication_url or "",
        "abstract": paper.abstract or "",
        "publication_date": paper.publication_date or "",
        "venue": paper.venue or "",
        "doi": paper.doi or "",
        "arxiv_id": paper.arxiv_id or "",
        "tags": "",
        "notes": "",
    }


async def _stage_uploaded_pdf(
    upload: UploadFile,
    *,
    directory: Path,
    max_bytes: int,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f".manual-upload-{uuid.uuid4().hex}.pdf.part"
    total = 0
    try:
        with temporary.open("xb") as handle:
            while True:
                chunk = await upload.read(64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise PDFError(f"PDF exceeds maximum size of {max_bytes} bytes")
                handle.write(chunk)
        if total == 0:
            raise PDFError("PDF file is empty")
        validate_pdf_file(temporary)
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


def _parse_filter_values(params: Any) -> PaperFilters:
    page = max(1, _int_or_none(params.get("page")) or 1)
    page_size = _int_or_none(params.get("page_size")) or DEFAULT_PAGE_SIZE
    if page_size not in PAGE_SIZES:
        page_size = DEFAULT_PAGE_SIZE
    paper_kind = params.get("kind", "")
    if paper_kind not in PAPER_KINDS:
        paper_kind = ""
    scope = params.get("scope", "")
    if scope not in SCOPES:
        scope = ""
    sort = params.get("sort", "newest")
    if sort not in SORTS:
        sort = "newest"
    raw_topics = params.getlist("topic") if hasattr(params, "getlist") else params.get("topic", ())
    if isinstance(raw_topics, str):
        raw_topics = (raw_topics,)
    topic_ids = tuple(dict.fromkeys(
        candidate for value in (raw_topics or ())
        if (candidate := short_openalex_id(str(value).strip()[:64]) or "").startswith("T")
    ))[:50]
    topic_id = topic_ids[0] if topic_ids else ""
    raw_authors = params.getlist("author") if hasattr(params, "getlist") else params.get("author", ())
    if isinstance(raw_authors, str):
        raw_authors = (raw_authors,)
    author_ids = tuple(dict.fromkeys(
        candidate for value in (raw_authors or ())
        if (candidate := short_openalex_id(str(value).strip()[:64]) or str(value).strip()[:64])
    ))[:50]
    author_id = author_ids[0] if author_ids else ""
    raw_author_tags = (
        params.getlist("author_tag")
        if hasattr(params, "getlist")
        else params.get("author_tag", ())
    )
    if isinstance(raw_author_tags, str):
        raw_author_tags = (raw_author_tags,)
    author_tags: list[str] = []
    seen_tags: set[str] = set()
    for value in raw_author_tags or ():
        tag = " ".join(str(value).strip().split())[:40]
        normalized = tag.casefold()
        if tag and normalized not in seen_tags:
            author_tags.append(tag)
            seen_tags.add(normalized)
        if len(author_tags) >= 50:
            break
    raw_content_tags = (
        params.getlist("tag")
        if hasattr(params, "getlist")
        else params.get("tag", ())
    )
    if isinstance(raw_content_tags, str):
        raw_content_tags = (raw_content_tags,)
    content_tags: list[str] = []
    seen_content_tags: set[str] = set()
    for value in raw_content_tags or ():
        tag = " ".join(str(value).strip().split())[:80]
        normalized = tag.casefold()
        if tag and normalized not in seen_content_tags:
            content_tags.append(tag)
            seen_content_tags.add(normalized)
        if len(content_tags) >= 50:
            break
    return PaperFilters(
        query=params.get("q", "").strip()[:200],
        title_query=params.get("title", "").strip()[:300],
        author_id=author_id,
        author_ids=author_ids,
        author_tags=tuple(author_tags),
        content_tags=tuple(content_tags),
        topic_id=topic_id,
        topic_ids=topic_ids,
        paper_kind=paper_kind,
        scope=scope,
        has_pdf=_parse_tristate(params.get("has_pdf")),
        analyzed=_parse_tristate(params.get("analyzed")),
        starred=_parse_tristate(params.get("starred")),
        read=_parse_tristate(params.get("read")),
        notes=_parse_tristate(params.get("notes")),
        display_stubs=_parse_tristate(params.get("display_stubs")) is True,
        date_from=_valid_date(params.get("date_from")),
        date_to=_valid_date(params.get("date_to")),
        sort=sort,
        page=page,
        page_size=page_size,
    )


def _parse_filters(request: Request) -> PaperFilters:
    return _parse_filter_values(request.query_params)


def _content_tag_values(params: Any) -> tuple[str, ...]:
    raw = params.getlist("tag") if hasattr(params, "getlist") else params.get("tag", ())
    if isinstance(raw, str):
        raw = (raw,)
    values: list[str] = []
    seen: set[str] = set()
    for value in raw or ():
        cleaned = " ".join(str(value).strip().split())[:80]
        key = cleaned.casefold()
        if cleaned and key not in seen:
            values.append(cleaned)
            seen.add(key)
    return tuple(values[:50])


def _split_tags(value: str) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for raw in re.split(r"[,;\n]", value):
        cleaned = " ".join(raw.strip().split())[:80]
        key = cleaned.casefold()
        if cleaned and key not in seen:
            values.append(cleaned)
            seen.add(key)
        if len(values) >= 100:
            break
    return tuple(values)


def _request_return_path(request: Request, fallback: str = "/") -> str:
    referer = urlparse(request.headers.get("referer", ""))
    if referer.path.startswith("/") and not referer.path.startswith("//"):
        return f"{referer.path}?{referer.query}" if referer.query else referer.path
    return fallback


def _async_request(request: Request) -> bool:
    return request.headers.get("X-Paperwatch-Async") == "1"


def _json_value(value: Any, fallback: Any) -> Any:
    if value in (None, ""):
        return fallback
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def _row_dict(row: Any) -> dict[str, Any]:
    return dict(row) if row is not None else {}


def _query_profile_names(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return (stripped,)
    else:
        decoded = value
    if isinstance(decoded, str):
        return (decoded.strip(),) if decoded.strip() else ()
    if not isinstance(decoded, list):
        return ()
    return tuple(dict.fromkeys(
        str(item).strip() for item in decoded if str(item).strip()
    ))


def _configured_context_window(
    config: AppConfig,
    profile_names: Iterable[str],
    requested: int | None = None,
) -> int:
    profiles = [
        profile
        for name in dict.fromkeys(str(value).strip() for value in profile_names)
        if name and (profile := config.llm_profile(name)) is not None
    ]
    native = [
        profile for profile in profiles if profile.provider in {"codex", "claude-code"}
    ]
    if native:
        if len(profiles) != 1:
            raise ValueError("Codex and Claude Code must run as the only selected model")
        return native[0].context_window_tokens
    if requested is not None:
        return requested
    if len(profiles) == 1:
        return profiles[0].context_window_tokens
    return config.llm.context_window_tokens


def _query_view(row: Any, config: AppConfig) -> dict[str, Any]:
    query = _row_dict(row)
    available = {item.profile: item for item in config.available_llms}
    selected = _query_profile_names(
        query.get("model_profile") or query.get("model_profiles_json")
    )[:1]
    if not selected and available:
        selected = (next(iter(available)),)
    query["model_profiles"] = list(selected)
    query["selected_models"] = [
        {
            "profile": name,
            "model": available[name].model if name in available else "Unavailable",
            "provider": available[name].provider if name in available else "missing",
        }
        for name in selected
    ]
    query["paper_source_scope"] = str(query.get("paper_source_scope") or "auto")
    query["pdf_strategy"] = str(query.get("pdf_strategy") or "chunked")
    query["regexp_pattern"] = str(query.get("regexp_pattern") or "")
    query["regexp_radius"] = int(query.get("regexp_radius") or 0)
    query["regexp_debug_text"] = str(query.get("regexp_debug_text") or "")
    query["regexp_mode"] = str(query.get("regexp_mode") or "query")
    query["content_scope"] = str(query.get("content_scope") or "both")
    query["query_kind"] = str(query.get("query_kind") or "summary")
    query["label_class"] = str(query.get("label_class") or "")
    label_values = _json_value(query.get("label_values_json"), [])
    if not isinstance(label_values, list):
        label_values = []
    query["label_values"] = [
        {
            "value": str(item.get("value") or ""),
            "pattern": str(item.get("pattern") or ""),
        }
        for item in label_values
        if isinstance(item, dict) and item.get("value") and item.get("pattern")
    ]
    query["label_values_text"] = "\n".join(
        f"{item['value']} = {item['pattern']}" for item in query["label_values"]
    )
    query["label_debug_text"] = str(query.get("label_debug_text") or "")
    query["local_regexp"] = bool(
        query["query_kind"] == "regexp"
        or (query["regexp_pattern"] and query["regexp_mode"] == "print")
    )
    return query


def _classification_form_values(value: str) -> list[dict[str, str]]:
    values: list[dict[str, str]] = []
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError("Each classification value must use: Label = regexp")
        label, pattern = line.split("=", 1)
        values.append({"value": label.strip(), "pattern": pattern.strip()})
    return values


def _query_execution_form(
    form: Any,
    config: AppConfig,
    *,
    existing_query: Any | None = None,
) -> dict[str, Any]:
    available = {item.profile: item for item in config.available_llms}

    def saved_text(name: str) -> str:
        if name in form:
            return str(form.get(name, ""))
        if existing_query is None:
            return ""
        try:
            return str(existing_query[name] or "")
        except (KeyError, TypeError, IndexError):
            return str(getattr(existing_query, name, "") or "")
    profiles = tuple(dict.fromkeys(
        str(value).strip() for value in form.getlist("model") if str(value).strip()
    ))
    unknown = [name for name in profiles if name not in available]
    if unknown:
        raise ValueError(f"Unknown configured model profile(s): {', '.join(unknown)}")
    if len(profiles) > 1:
        raise ValueError("A saved query can use only one model")
    if available and not profiles:
        profiles = (next(iter(available)),)
    source_scope = str(form.get("paper_source_scope", "auto"))
    pdf_strategy = str(form.get("pdf_strategy", "chunked"))
    if source_scope in {"abstract", "pdf"}:
        pdf_strategy = "auto"
    regexp_radius = _int_or_none(form.get("regexp_radius"))
    if regexp_radius is None:
        regexp_radius = 0
    query_kind = str(form.get("query_kind", "summary"))
    if query_kind == "regexp":
        profiles = ()
        regexp_pattern = ""
        regexp_radius = 0
        regexp_mode = "query"
    else:
        regexp_pattern = str(form.get("regexp_pattern", ""))
        regexp_mode = str(form.get("regexp_mode", "query"))
    if source_scope == "pdf":
        profile = available.get(profiles[0]) if profiles else None
        if profile is None or profile.provider != "codex":
            raise ValueError("Whole-PDF upload requires a Codex model")
    return {
        "model_profiles": profiles,
        "paper_source_scope": source_scope,
        "pdf_strategy": pdf_strategy,
        "regexp_pattern": regexp_pattern,
        "regexp_radius": regexp_radius,
        "regexp_debug_text": saved_text("regexp_debug_text"),
        "regexp_mode": regexp_mode,
        "content_scope": str(form.get("content_scope", "both")),
        "query_kind": query_kind,
        "label_class": str(form.get("label_class", "")),
        "label_values": (
            _classification_form_values(str(form.get("label_values", "")))
            if query_kind in {"classification", "regexp"}
            else []
        ),
        "label_debug_text": saved_text("label_debug_text"),
    }


def _query_supports(query: Any, content_type: str) -> bool:
    scope = str(query.get("content_scope", "both") if isinstance(query, dict) else query["content_scope"] or "both")
    return scope == "both" or scope == content_type


def _saved_query_run_options(
    form: Any,
    query: Any,
    config: AppConfig,
    *,
    content_type: str,
) -> tuple[dict[str, Any], tuple[str, ...], int, str, str]:
    """Resolve a saved query's defaults unless this run explicitly overrides them."""
    query_view = _query_view(query, config)
    override_defaults = str(form.get("override_query_defaults", "")) in {
        "1", "true", "on"
    }
    if override_defaults:
        profiles = tuple(dict.fromkeys(
            str(value).strip()
            for value in form.getlist("model")
            if str(value).strip()
        ))
        requested_context = _int_or_none(form.get("context_window_tokens"))
        source_scope = str(
            form.get("source_scope", query_view["paper_source_scope"])
        )
        pdf_strategy = str(form.get("pdf_strategy", query_view["pdf_strategy"]))
    else:
        profiles = tuple(query_view["model_profiles"])
        requested_context = None
        source_scope = str(query_view["paper_source_scope"])
        pdf_strategy = str(query_view["pdf_strategy"])

    if content_type == "blogs":
        source_scope = "blog"
        pdf_strategy = "auto"
    elif content_type != "papers":
        raise ValueError("Content type must be papers or blogs")

    local_regexp = bool(query_view["local_regexp"])
    if local_regexp:
        profiles = ()
    available_profiles = {item.profile for item in config.available_llms}
    if not local_regexp and (
        not profiles or any(name not in available_profiles for name in profiles)
    ):
        raise ValueError("Select a configured model.")
    context = _configured_context_window(config, profiles, requested_context)
    return query_view, profiles, context, source_scope, pdf_strategy


def _analysis_view(row: Any) -> dict[str, Any]:
    data = _row_dict(row)
    result = _json_value(data.get("result_json"), {})
    data["result"] = result
    data["usage"] = _json_value(data.get("usage_json"), {})
    return data


def _paper_export(db: Database, row: Any) -> dict[str, Any]:
    paper = _row_dict(row)
    paper["metadata"] = _json_value(paper.pop("metadata_json", "{}"), {})
    exported_authors = []
    for item in db.paper_authors(int(paper["id"])):
        author = _row_dict(item)
        author.pop("tracked_enabled", None)
        author.pop("is_tracked", None)
        exported_authors.append(author)
    paper["authors"] = exported_authors
    paper["topics"] = [
        {
            "openalex_id": item["openalex_id"],
            "display_name": item["display_name"],
            "score": item["score"],
            "is_primary": bool(item["is_primary"]),
            "subfield": {
                "id": item["subfield_id"],
                "display_name": item["subfield_name"],
            },
            "field": {
                "id": item["field_id"],
                "display_name": item["field_name"],
            },
            "domain": {
                "id": item["domain_id"],
                "display_name": item["domain_name"],
            },
        }
        for item in db.paper_topics(int(paper["id"]))
    ]
    paper["analyses"] = [
        _json_value(item["result_json"], {}) | {
            "analysis_id": item["id"],
            "scope": item["scope"],
            "provider": item["provider"],
            "model": item["model"],
            "prompt_version": item["prompt_version"],
            "created_at": item["created_at"],
            "usage": _json_value(item["usage_json"], {}),
        }
        for item in db.paper_analyses(int(paper["id"]))
    ]
    for key in (
        "author_names", "topics_json", "analysis_count", "latest_scope",
        "latest_kind", "latest_summary"
    ):
        paper.pop(key, None)
    return paper


def _all_filtered(
    db: Database, filters: PaperFilters, *, include_dead: bool = False
) -> list[Any]:
    rows, _ = db.search_papers(
        query=filters.query,
        title_query=filters.title_query,
        author_id=filters.author_id,
        author_ids=filters.author_ids,
        author_tags=filters.author_tags,
        content_tags=filters.content_tags,
        topic_id=filters.topic_id,
        topic_ids=filters.topic_ids,
        paper_kind=filters.paper_kind,
        scope=filters.scope,
        has_pdf=filters.has_pdf,
        analyzed=filters.analyzed,
        date_from=filters.date_from,
        date_to=filters.date_to,
        sort=filters.sort,
        limit=None,
        offset=0,
        include_dead=include_dead,
    )
    return rows


def _content_kind(value: str | None, default: str = "all") -> str:
    normalized = str(value or default).strip().lower()
    return normalized if normalized in {"all", "papers", "blogs"} else default


def _website_view(value: str) -> tuple[str, str]:
    parsed = urlparse(str(value or ""))
    host = (parsed.hostname or "").lower()
    display = host.removeprefix("www.") or str(value or "")
    if parsed.scheme.lower() in {"http", "https"} and parsed.netloc:
        return display, f"{parsed.scheme.lower()}://{parsed.netloc}/"
    return display, str(value or "")


def _record_sort_key(record: dict[str, Any], sort: str) -> tuple[Any, ...]:
    if sort == "title":
        return (str(record.get("title") or "").casefold(), int(record.get("id") or 0))
    date_value = str(record.get("record_date") or "")
    if sort == "oldest":
        return (date_value or "9999-12-31", int(record.get("id") or 0))
    if sort == "discovered":
        return (str(record.get("record_fetched") or ""), int(record.get("id") or 0))
    return (date_value, int(record.get("id") or 0))


def _safe_markdown_url(value: str) -> str | None:
    parsed = urlparse(html.unescape(value).strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None
    return parsed.geturl()


def _render_stored_markdown(value: str) -> str:
    """Render stored summaries and extracted blog markdown without raw HTML.

    Remote images remain URLs. Browsers fetch them only while the detail page is open;
    Paperwatch never stores the image bytes.
    """
    image_pattern = re.compile(r"!\[([^\]]*)\]\((https?://[^\s\)]+)\)")
    link_pattern = re.compile(r"(?<!!)\[([^\]]+)\]\((https?://[^\s\)]+)\)")

    def image(match: re.Match[str]) -> str:
        url = _safe_markdown_url(match.group(2))
        if url is None:
            return match.group(0)
        alt = html.escape(html.unescape(match.group(1)) or "Image")
        safe_url = html.escape(url, quote=True)
        return (
            '<figure class="remote-blog-image"><a href="' + safe_url + '" '
            'target="_blank" rel="noreferrer noopener"><img src="' + safe_url + '" '
            'alt="' + alt + '" loading="lazy" decoding="async" '
            'referrerpolicy="no-referrer"></a><figcaption><a href="' + safe_url + '" '
            'target="_blank" rel="noreferrer noopener">Open image</a></figcaption></figure>'
        )

    def link(match: re.Match[str]) -> str:
        url = _safe_markdown_url(match.group(2))
        if url is None:
            return match.group(0)
        label = html.escape(html.unescape(match.group(1)))
        return (
            '<a href="' + html.escape(url, quote=True) + '" target="_blank" '
            'rel="noreferrer noopener">' + label + '</a>'
        )

    def inline(raw: str) -> str:
        escaped = html.escape(raw)
        code_values: list[str] = []

        def code(match: re.Match[str]) -> str:
            code_values.append(f"<code>{match.group(1)}</code>")
            return f"\x00CODE{len(code_values) - 1}\x00"

        escaped = re.sub(r"`([^`\n]+)`", code, escaped)
        escaped = image_pattern.sub(image, escaped)
        escaped = link_pattern.sub(link, escaped)
        escaped = re.sub(
            r"!?\[([^\]]*)\]\((?!https?://)[^\)]*\)",
            lambda match: html.escape(html.unescape(match.group(1))),
            escaped,
            flags=re.IGNORECASE,
        )
        escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
        escaped = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", escaped)
        for index, rendered in enumerate(code_values):
            escaped = escaped.replace(f"\x00CODE{index}\x00", rendered)
        return escaped

    lines = (value or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[str] = []
    paragraph: list[str] = []
    list_items: list[str] = []
    list_kind = "ul"
    quote_lines: list[str] = []
    code_lines: list[str] = []
    in_code = False

    def flush_paragraph() -> None:
        if paragraph:
            rendered = "<br>".join(inline(line) for line in paragraph)
            if rendered.startswith("<figure") and rendered.endswith("</figure>"):
                blocks.append(rendered)
            else:
                blocks.append(f"<p>{rendered}</p>")
            paragraph.clear()

    def flush_list() -> None:
        if list_items:
            blocks.append(
                f"<{list_kind}>" + "".join(f"<li>{inline(item)}</li>" for item in list_items) + f"</{list_kind}>"
            )
            list_items.clear()

    def flush_quote() -> None:
        if quote_lines:
            blocks.append(f"<blockquote>{'<br>'.join(inline(line) for line in quote_lines)}</blockquote>")
            quote_lines.clear()

    for line in lines:
        if line.strip().startswith("```"):
            flush_paragraph()
            flush_list()
            flush_quote()
            if in_code:
                blocks.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
                code_lines.clear()
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not line.strip():
            flush_paragraph()
            flush_list()
            flush_quote()
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            flush_list()
            flush_quote()
            level = len(heading.group(1))
            blocks.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
            continue
        unordered = re.match(r"^\s*[-*+]\s+(.+)$", line)
        ordered = re.match(r"^\s*\d+[.)]\s+(.+)$", line)
        if unordered or ordered:
            flush_paragraph()
            flush_quote()
            kind = "ol" if ordered else "ul"
            if list_items and kind != list_kind:
                flush_list()
            list_kind = kind
            list_items.append((ordered or unordered).group(1))
            continue
        if line.lstrip().startswith(">"):
            flush_paragraph()
            flush_list()
            quote_lines.append(line.lstrip()[1:].lstrip())
            continue
        flush_list()
        flush_quote()
        paragraph.append(line)
    if in_code and code_lines:
        blocks.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
    flush_paragraph()
    flush_list()
    flush_quote()
    return "".join(blocks)


def _notice_url(path: str, message: str, *, level: str = "success") -> str:
    separator = "&" if "?" in path else "?"
    return f"{path}{separator}{urlencode({'notice': message, 'level': level})}"


def _author_client(config: AppConfig) -> OpenAlexClient:
    return OpenAlexClient(
        api_key=config.openalex.api_key,
        base_url=config.openalex.base_url,
        timeout_seconds=config.openalex.timeout_seconds,
    )


def _arxiv_client(config: AppConfig) -> ArxivClient:
    return ArxivClient(
        base_url=config.arxiv.base_url,
        timeout_seconds=config.arxiv.timeout_seconds,
        title_match_threshold=config.arxiv.title_match_threshold,
        min_interval_seconds=config.arxiv.min_interval_seconds,
    )


def _blog_client(_config: AppConfig) -> httpx.Client:
    return httpx.Client(
        timeout=60,
        follow_redirects=True,
        headers={"User-Agent": "paperwatch/0.10.2 (+local discovery)"},
    )


def _paper_from_arxiv(entry: ArxivEntry) -> PaperRecord:
    authors = [
        PaperAuthor(
            openalex_id=None,
            name=name,
            position=(
                "first"
                if index == 0
                else "last"
                if index == len(entry.authors) - 1
                else "middle"
            ),
        )
        for index, name in enumerate(entry.authors)
    ]
    return PaperRecord(
        openalex_id=None,
        doi=entry.doi,
        arxiv_id=entry.arxiv_id,
        title=entry.title,
        abstract=entry.abstract,
        publication_date=entry.published,
        updated_date=entry.updated,
        venue="arXiv",
        publication_url=entry.abstract_url,
        pdf_url=entry.pdf_url,
        abstract_source="arxiv",
        pdf_source="arxiv" if entry.pdf_url else None,
        authors=authors,
        metadata={"arxiv_search": True},
    )


def _search_arxiv_identifier(
    query: str,
    *,
    openalex: OpenAlexClient,
    arxiv: ArxivClient,
    limit: int,
) -> list[PaperRecord]:
    arxiv_id = extract_arxiv_id(query)
    if not arxiv_id:
        return openalex.search_works(query, limit=limit)

    # OpenAlex documents DOI, rather than arXiv id, as a singleton external
    # work identifier. Try the arXiv-issued DataCite DOI first.
    try:
        return [openalex.get_work_by_doi(f"10.48550/arXiv.{arxiv_id}")]
    except Exception:
        pass

    entry = arxiv.by_id(arxiv_id)
    if entry is None:
        return []

    # Older or not-yet-linked OpenAlex records may not resolve via the arXiv
    # DOI. Use arXiv only to resolve the identifier to a title, then search
    # OpenAlex by that title. Prefer an exact title match if one exists.
    try:
        candidates = openalex.search_works(entry.title, limit=limit)
    except Exception:
        candidates = []
    target = normalize_title(entry.title)
    exact = [paper for paper in candidates if normalize_title(paper.title) == target]
    if exact:
        others = [paper for paper in candidates if normalize_title(paper.title) != target]
        return [*exact, *others]
    # Discover is intentionally an OpenAlex search surface. arXiv resolves the
    # user's identifier to metadata, but only OpenAlex work records are shown.
    return candidates


def _discover_paper_result(db: Database, paper: PaperRecord) -> dict[str, Any]:
    stored = db.find_paper(paper)
    stored_id = int(stored["id"]) if stored is not None else None
    local_path = str(stored["local_pdf_path"] or "").strip() if stored is not None else ""
    downloaded = bool(local_path and Path(local_path).expanduser().is_file())
    return {
        "paper": paper,
        "openalex_id": paper.openalex_id or "",
        "stored_id": stored_id,
        "downloaded": downloaded,
        "local_url": f"/papers/{stored_id}" if downloaded and stored_id is not None else "",
    }


def _blog_preview_entries(fetched: BlogFetch, source_url: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    articles = {item.url: item for item in fetched.article_items}

    for item in fetched.feed_items:
        url = item.url or source_url
        if url in seen:
            continue
        seen.add(url)
        article = articles.get(url)
        text = article.text if article is not None else item.text
        entries.append(
            {
                "title": article.title if article is not None else item.title,
                "url": url,
                "published_at": (
                    article.published_at if article is not None else item.published_at
                ),
                "summary": clean_text(text)[:700] if text else "",
            }
        )

    for item in fetched.article_items:
        if item.url in seen:
            continue
        seen.add(item.url)
        entries.append(
            {
                "title": item.title,
                "url": item.url,
                "published_at": item.published_at,
                "summary": clean_text(item.text)[:700],
            }
        )

    for item in fetched.discovered_articles:
        if item.url in seen:
            continue
        seen.add(item.url)
        entries.append(
            {
                "title": item.title,
                "url": item.url,
                "published_at": item.published_at,
                "summary": clean_text(item.summary)[:700],
            }
        )

    if not entries and fetched.snapshot_text:
        entries.append(
            {
                "title": fetched.title or source_url,
                "url": source_url,
                "published_at": fetched.snapshot_published_at,
                "summary": clean_text(fetched.snapshot_text)[:700],
            }
        )
    return entries


def _fetch_blog_preview(client: httpx.Client, url: str, *, source_mode: str = "auto") -> BlogFetch:
    normalized = normalize_blog_url(url)
    return fetch_blog(
        client,
        {
            "url": normalized,
            "title": normalized,
            "source_mode": source_mode,
            "feed_url": None,
            "feed_etag": None,
            "feed_last_modified": None,
            "feed_discovery_checked": False,
            "content_selector": None,
            "article_url_pattern": None,
            "article_cursor": 0,
            "_known_article_urls": (),
            "_aggressive_discovery": source_mode != "article",
            "_feed_candidate_limit": 16,
            "_article_fetch_limit": 24,
        },
    )


def _save_discovered_blog_fetch(
    db: Database,
    *,
    author_openalex_id: str,
    source_url: str,
    fetched: BlogFetch,
    label: str = "",
    relationship: str = "",
    tags: tuple[str, ...] = (),
    notes: str = "",
) -> tuple[int, int]:
    normalized_source = normalize_blog_url(source_url)
    blog_id = db.add_author_blog(
        author_openalex_id,
        normalized_source,
        label=label or fetched.title,
        relationship=relationship,
        source_mode=fetched.source_mode or infer_source_mode(normalized_source),
    )
    for tag in tags:
        db.add_blog_source_tag(blog_id, tag)
    if notes.strip():
        db.set_object_notes("blog_source", blog_id, notes)

    saved = 0
    seen_urls: set[str] = set()
    article_by_url = {item.url: item for item in fetched.article_items}
    for item in fetched.feed_items:
        article = article_by_url.get(item.url or "")
        text = article.text if article is not None else item.text
        rendered_html = article.rendered_html if article is not None else item.rendered_html
        sha256 = article.sha256 if article is not None else item.sha256
        published_at = article.published_at if article is not None else item.published_at
        source = item.url or normalized_source
        previous = db.latest_blog_entry(
            blog_id, entry_kind="rss", external_id=item.external_id
        )
        if previous is not None and str(previous["content_sha256"]) == sha256:
            seen_urls.add(source)
            continue
        db.save_blog_entry(
            blog_id=blog_id,
            title=item.title,
            content_sha256=sha256,
            normalized_text=text,
            diff_text="",
            change_kind="changed",
            previous_entry_id=int(previous["id"]) if previous is not None else None,
            entry_kind="rss",
            external_id=item.external_id,
            source_url=source,
            published_at=published_at,
            rendered_html=rendered_html,
        )
        seen_urls.add(source)
        saved += 1

    for item in fetched.article_items:
        if item.url in seen_urls:
            continue
        previous = db.latest_blog_entry_for_source(
            blog_id, entry_kind="snapshot", source_url=item.url
        )
        if previous is not None and str(previous["content_sha256"]) == item.sha256:
            continue
        db.save_blog_entry(
            blog_id=blog_id,
            title=item.title,
            content_sha256=item.sha256,
            normalized_text=item.text,
            diff_text="",
            change_kind="changed",
            previous_entry_id=int(previous["id"]) if previous is not None else None,
            entry_kind="snapshot",
            external_id=item.external_id,
            source_url=item.url,
            published_at=item.published_at,
            rendered_html=item.rendered_html,
        )
        seen_urls.add(item.url)
        saved += 1

    fetched_urls = {item.url for item in fetched.article_items}
    for item in fetched.discovered_articles:
        if item.url in seen_urls or item.url in fetched_urls:
            continue
        external_id = hashlib.sha256(
            f"discover-blog\0{item.url}".encode("utf-8")
        ).hexdigest()
        previous = db.latest_blog_entry(
            blog_id, entry_kind="rss", external_id=external_id
        )
        if previous is not None:
            continue
        text = "\n".join(
            value
            for value in (
                f"Title: {item.title}",
                f"URL: {item.url}",
                item.summary,
            )
            if value
        )
        db.save_blog_entry(
            blog_id=blog_id,
            title=item.title,
            content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            normalized_text=text,
            diff_text="",
            change_kind="changed",
            previous_entry_id=None,
            entry_kind="rss",
            external_id=external_id,
            source_url=item.url,
            published_at=item.published_at,
        )
        saved += 1

    if not fetched.feed_items and not fetched.article_items and fetched.snapshot_text:
        sha256 = fetched.snapshot_sha256 or hashlib.sha256(
            fetched.snapshot_text.encode("utf-8")
        ).hexdigest()
        external_id = hashlib.sha256(
            f"discover-snapshot\0{normalized_source}".encode("utf-8")
        ).hexdigest()
        previous = db.latest_blog_entry(
            blog_id, entry_kind="snapshot", external_id=external_id
        )
        if previous is None or str(previous["content_sha256"]) != sha256:
            db.save_blog_entry(
                blog_id=blog_id,
                title=fetched.title or normalized_source,
                content_sha256=sha256,
                normalized_text=fetched.snapshot_text,
                diff_text="",
                change_kind="changed",
                previous_entry_id=int(previous["id"]) if previous is not None else None,
                entry_kind="snapshot",
                external_id=external_id,
                source_url=normalized_source,
                published_at=fetched.snapshot_published_at,
                rendered_html=fetched.snapshot_rendered_html,
            )
            saved += 1

    db.mark_blog_checked(
        blog_id,
        status=fetched.status_code,
        etag=fetched.etag,
        last_modified=fetched.last_modified,
        title=fetched.title,
        error=None,
        fetch_method=fetched.method,
        feed_url=fetched.feed_url,
        feed_initialized=True if fetched.method == "rss" else None,
        feed_discovery_checked=fetched.discovery_checked,
        article_initialized=True if fetched.article_items else None,
        article_cursor=fetched.article_cursor_next,
        diagnostics=json.dumps(fetched.diagnostics),
        item_stream_mode=fetched.source_mode,
    )
    return blog_id, saved


def _author_view(row: Any) -> dict[str, Any]:
    data = _row_dict(row)
    name = str(data.get("name") or "Unknown author")
    author_id = str(data.get("openalex_author_id") or data.get("openalex_id") or "")
    scholar_query = quote_plus(f'author:"{name}"')
    encoded_name = quote_plus(name)
    data.update(
        {
            "author_id": author_id,
            "papers_url": (
                f"/?author={quote_plus(author_id)}"
                if author_id
                else f"/?q={encoded_name}"
            ),
            "author_url": f"/authors/{quote_plus(author_id)}/blogs" if author_id else "/authors",
            "google_scholar_url": f"https://scholar.google.com/scholar?q={scholar_query}",
            "openalex_url": (f"https://openalex.org/authors/{author_id.lower()}" if author_id else None),
            "arxiv_url": f"https://arxiv.org/search/?query={encoded_name}&searchtype=author",
        }
    )
    return data


def _paper_view(row: Any) -> dict[str, Any]:
    data = _row_dict(row)
    topics = _json_value(data.get("topics_json"), [])
    data["topics"] = topics if isinstance(topics, list) else []
    title = str(data.get("title") or "")
    data["google_scholar_url"] = (
        f"https://scholar.google.com/scholar?q={quote_plus('\"' + title + '\"')}"
        if title else None
    )
    openalex_id = str(data.get("openalex_id") or "")
    data["openalex_url"] = (
        f"https://openalex.org/works/{openalex_id.lower()}" if openalex_id else None
    )
    arxiv_id = str(data.get("arxiv_id") or "")
    data["arxiv_url"] = f"https://arxiv.org/abs/{quote_plus(arxiv_id)}" if arxiv_id else None
    return data


def _paper_pdf_state(
    db: Database, data: dict[str, Any], *, verify_hash: bool = True
) -> dict[str, Any]:
    value = str(data.get("local_pdf_path") or "").strip()
    known_modified = bool(data.get("pdf_modified_on_disk"))
    data["pdf_modified"] = known_modified
    data["pdf_missing_on_disk"] = False
    if not value:
        return data
    path = Path(value).expanduser()
    try:
        stat = path.stat()
    except OSError:
        data["pdf_modified"] = True
        data["pdf_missing_on_disk"] = True
        return data
    size = int(stat.st_size)
    mtime_ns = int(stat.st_mtime_ns)
    stored_size = data.get("pdf_size")
    stored_mtime = data.get("pdf_mtime_ns")

    # A matching cheap fingerprint makes list rendering constant-time. Legacy rows
    # without one are verified only on the detail page rather than hashing every PDF
    # whenever Content is opened.
    if stored_size is not None and stored_mtime is not None:
        fingerprint_changed = int(stored_size) != size or int(stored_mtime) != mtime_ns
        if not fingerprint_changed:
            return data
    elif not verify_hash:
        return data

    if not verify_hash:
        data["pdf_modified"] = True
        return data
    try:
        current = pdf_sha256(path)
    except PDFError:
        data["pdf_modified"] = True
        return data
    stored = str(data.get("pdf_sha256") or "")
    if not stored:
        db.set_paper_pdf_hash(
            int(data["id"]), current, pdf_size=size, pdf_mtime_ns=mtime_ns
        )
        data["pdf_sha256"] = current
        data["pdf_modified_on_disk"] = 0
        data["pdf_modified"] = False
    elif stored != current:
        data["pdf_modified"] = True
        data["pdf_modified_on_disk"] = 1
        db.set_paper_pdf_fingerprint(
            int(data["id"]), pdf_size=size, pdf_mtime_ns=mtime_ns, modified=True
        )
    else:
        db.set_paper_pdf_fingerprint(
            int(data["id"]), pdf_size=size, pdf_mtime_ns=mtime_ns, modified=False
        )
        data["pdf_size"] = size
        data["pdf_mtime_ns"] = mtime_ns
        data["pdf_modified_on_disk"] = 0
        data["pdf_modified"] = False
    data["current_pdf_sha256"] = current
    return data


def _paper_table_view(db: Database, row: Any) -> dict[str, Any]:
    data = _paper_pdf_state(db, _paper_view(row), verify_hash=False)
    tracked = _json_value(data.get("tracked_authors_json"), [])
    if not isinstance(tracked, list):
        tracked = []
    data["tracked_authors"] = [
        {
            "name": str(item.get("name") or "Unknown author"),
            "author_url": (
                f"/authors/{quote_plus(str(item.get('openalex_id') or ''))}/blogs"
                if item.get("openalex_id") else "/authors"
            ),
        }
        for item in tracked
        if isinstance(item, dict)
    ]
    author_count = int(data.get("author_count") or 0)
    data["other_author_count"] = max(0, author_count - len(data["tracked_authors"]))
    data["all_authors"] = []
    return data

def _selection_with_labels(
    filters: PaperFilters,
    *,
    authors: Iterable[Any],
    topics: Iterable[Any],
) -> dict[str, Any]:
    selection = filters.selection()
    labels: list[str] = []
    if filters.query:
        labels.append(f'Search “{filters.query}”')
    if filters.author_id:
        name = next(
            (str(row["name"]) for row in authors if row["openalex_id"] == filters.author_id),
            filters.author_id,
        )
        labels.append(f"Author: {name}")
    if filters.author_tags:
        author_names = {
            str(row["openalex_id"]): str(row["name"])
            for row in authors
            if row["openalex_id"]
        }
        selected_authors = [
            author_names.get(short_openalex_id(value) or value, value)
            for value in filters.author_tags
        ]
        labels.append(f"Authors: {', '.join(selected_authors)}")
    if filters.topic_id:
        name = next(
            (
                str(row["display_name"])
                for row in topics
                if row["openalex_id"] == filters.topic_id
            ),
            filters.topic_id,
        )
        labels.append(f"Topic: {name}")
    if filters.paper_kind:
        labels.append(f"Kind: {filters.paper_kind.replace('_', ' ')}")
    if filters.scope:
        labels.append(f"Scope: {filters.scope}")
    if filters.has_pdf is not None:
        labels.append("PDF available" if filters.has_pdf else "PDF missing")
    if filters.analyzed is not None:
        labels.append("Analyzed" if filters.analyzed else "Not analyzed")
    if filters.date_from:
        labels.append(f"From {filters.date_from}")
    if filters.date_to:
        labels.append(f"To {filters.date_to}")
    selection["labels"] = labels
    return selection


def _query_output(row: Any) -> dict[str, Any]:
    data = _row_dict(row)
    data["output"] = _json_value(data.get("output_json"), None)
    selection = _json_value(data.get("selection_json"), {})
    data["selection"] = selection if isinstance(selection, dict) else {}
    selection_items = []
    labels = data["selection"].get("labels")
    if isinstance(labels, list):
        selection_items.extend(str(item) for item in labels if item)
    if not selection_items:
        query = str(data["selection"].get("query") or "")
        if query:
            selection_items.append(f'Search “{query}”')
        if data["selection"].get("date_from"):
            selection_items.append(f'From {data["selection"]["date_from"]}')
        if data["selection"].get("date_to"):
            selection_items.append(f'To {data["selection"]["date_to"]}')
    content_type = str(data["selection"].get("content_type") or "papers")
    source_scope = str(
        data["selection"].get("source_scope")
        or ("blog" if content_type == "blogs" else "abstract")
    )
    pdf_strategy = str(data["selection"].get("pdf_strategy") or "auto")
    selection_items.append(f"Source: {source_scope}")
    if source_scope == "fulltext":
        selection_items.append(f"PDF strategy: {pdf_strategy}")
    data["selection_label"] = " · ".join(selection_items) or "All matching records"
    raw_tags = data["selection"].get("author_tags")
    author_tags = raw_tags if isinstance(raw_tags, list) else []
    selection_pairs = [
        ("q", data["selection"].get("query")),
        ("author", data["selection"].get("author_id")),
        *(("author_tag", tag) for tag in author_tags if tag),
        ("topic", data["selection"].get("topic_id")),
        ("kind", data["selection"].get("paper_kind")),
        ("scope", data["selection"].get("scope")),
        ("has_pdf", _bool_query(data["selection"].get("has_pdf"))),
        ("analyzed", _bool_query(data["selection"].get("analyzed"))),
        ("date_from", data["selection"].get("date_from")),
        ("date_to", data["selection"].get("date_to")),
        (
            "sort",
            data["selection"].get("sort")
            if data["selection"].get("sort") != "newest"
            else "",
        ),
    ]
    selection_query = urlencode([(key, value) for key, value in selection_pairs if value])
    if content_type == "blogs":
        data["selection_url"] = "/blogs"
    else:
        data["selection_url"] = f"/?{selection_query}" if selection_query else "/"
    return data


def _csv_response(records: Iterable[dict[str, Any]]) -> Response:
    stream = io.StringIO(newline="")
    fields = [
        "id",
        "title",
        "publication_date",
        "authors",
        "topics",
        "venue",
        "doi",
        "arxiv_id",
        "openalex_id",
        "dead",
        "abstract",
        "publication_url",
        "pdf_url",
        "local_pdf_path",
        "analysis_scope",
        "paper_kind",
        "one_sentence_summary",
        "abstract_summary",
        "research_problem",
        "key_contributions",
        "limitations_and_open_questions",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for record in records:
        author_names = [item.get("name", "") for item in record.get("authors", [])]
        analyses = record.get("analyses") or [None]
        for analysis in analyses:
            analysis = analysis or {}
            writer.writerow(
                {
                    **record,
                    "authors": "; ".join(author_names),
                    "topics": "; ".join(
                        item.get("display_name", item.get("name", ""))
                        for item in record.get("topics", [])
                    ),
                    "analysis_scope": analysis.get("scope", ""),
                    "paper_kind": analysis.get("paper_kind", ""),
                    "one_sentence_summary": analysis.get("one_sentence_summary", ""),
                    "abstract_summary": analysis.get("abstract_summary", ""),
                    "research_problem": analysis.get("research_problem", ""),
                    "key_contributions": json.dumps(
                        analysis.get("key_contributions", []), ensure_ascii=False
                    ),
                    "limitations_and_open_questions": json.dumps(
                        analysis.get("limitations_and_open_questions", []), ensure_ascii=False
                    ),
                }
            )
    return Response(
        stream.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="paperwatch-papers.csv"'},
    )


def create_app(
    config_path: str | Path = "config.toml",
    *,
    run_controller: RunController | None = None,
    analysis_controller: AnalysisController | None = None,
    query_controller: QueryController | None = None,
    author_client_factory: Callable[[AppConfig], Any] | None = None,
    arxiv_client_factory: Callable[[AppConfig], Any] | None = None,
    blog_client_factory: Callable[[AppConfig], Any] | None = None,
) -> FastAPI:
    config = load_config(config_path)
    controller = run_controller or RunController(config.config_path)
    model_controller = analysis_controller or AnalysisController(config.config_path)
    collection_controller = query_controller or QueryController(config.config_path)
    client_factory = author_client_factory or _author_client
    arxiv_factory = arxiv_client_factory or _arxiv_client
    blog_factory = blog_client_factory or _blog_client
    try:
        with analysis_lock(config.paths.database):
            with Database(config.paths.database) as recovery_db:
                recovery_db.fail_incomplete_query_runs(
                    "Interrupted because the web application restarted"
                )
                recovery_db.fail_incomplete_analysis_tasks(
                    "Interrupted because the web application restarted"
                )
    except AnalysisAlreadyInProgress:
        pass
    package_dir = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=str(package_dir / "web_templates"))
    templates.env.filters["fromjson"] = lambda value: _json_value(value, {})
    templates.env.filters["prettyjson"] = lambda value: json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    )
    templates.env.filters["querystring"] = lambda items: urlencode(items)
    templates.env.filters["markdown"] = _render_stored_markdown

    app = FastAPI(title="paperwatch", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.config = config
    app.state.run_controller = controller
    app.state.analysis_controller = model_controller
    app.state.query_controller = collection_controller
    app.state.author_client_factory = client_factory
    app.state.blog_client_factory = blog_factory
    app.mount("/static", StaticFiles(directory=str(package_dir / "web_static")), name="static")

    def render(request: Request, template: str, **context: Any) -> HTMLResponse:
        base = {
            "request": request,
            "config": config,
            "asset_version": __version__,
            "run_state": controller.snapshot(),
            "analysis_state": model_controller.snapshot(),
            "query_state": collection_controller.snapshot(),
            "notice": request.query_params.get("notice", "")[:300],
            "notice_level": request.query_params.get("level", "success"),
            "paper_kinds": PAPER_KINDS,
            "scopes": SCOPES,
            "page_sizes": PAGE_SIZES,
        }
        return templates.TemplateResponse(request, template, {**base, **context})

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Callable[..., Any]) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self'; img-src 'self' data: https: http:; "
            "script-src 'self'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
        )
        return response

    def _render_content_page(
        request: Request,
        *,
        default_content: str = "all",
        is_home: bool = False,
    ) -> HTMLResponse:
        filters = _parse_filters(request)
        content_filter = _content_kind(request.query_params.get("content"), default_content)
        selected_query_id = _int_or_none(request.query_params.get("query_id"))
        offset = (filters.page - 1) * filters.page_size
        prefetch = offset + filters.page_size
        paper_filters = {
            "query": filters.query,
            "title_query": filters.title_query,
            "author_id": filters.author_id,
            "author_ids": filters.author_ids,
            "author_tags": filters.author_tags,
            "content_tags": filters.content_tags,
            "topic_id": filters.topic_id,
            "topic_ids": filters.topic_ids,
            "paper_kind": filters.paper_kind,
            "scope": filters.scope,
            "has_pdf": filters.has_pdf,
            "analyzed": filters.analyzed,
            "starred": filters.starred,
            "read": filters.read,
            "notes": filters.notes,
            "date_from": filters.date_from,
            "date_to": filters.date_to,
            "sort": filters.sort,
        }
        blog_filters = {
            "query": filters.query,
            "title_query": filters.title_query,
            "tags": filters.content_tags,
            "author_id": filters.author_id,
            "author_ids": filters.author_ids,
            "author_tags": filters.author_tags,
            "date_from": filters.date_from,
            "date_to": filters.date_to,
            "sort": filters.sort,
            "starred": filters.starred,
            "read": filters.read,
            "notes": filters.notes,
            "analyzed": filters.analyzed,
            "show_stubs": filters.display_stubs,
        }
        blog_compatible_filters = not (
            filters.topic_id
            or filters.topic_ids
            or filters.paper_kind
            or filters.scope
            or filters.has_pdf is True
        )

        with Database(config.paths.database) as db:
            paper_rows: list[Any] = []
            blog_rows: list[Any] = []
            paper_total = 0
            blog_total = 0
            selected_matching_papers = 0
            selected_matching_blogs = 0
            has_blog_entries = db.has_visible_blog_entries()
            if content_filter != "blogs":
                paper_rows, paper_total = db.search_papers(
                    **paper_filters,
                    selected=False,
                    limit=(filters.page_size if content_filter == "papers" else prefetch),
                    offset=(offset if content_filter == "papers" else 0),
                    table_view=True,
                )
                _, selected_matching_papers = db.search_papers(
                    **paper_filters, selected=True, limit=0, offset=0
                )
            if (
                has_blog_entries
                and content_filter != "papers"
                and blog_compatible_filters
            ):
                blog_total = db.count_blog_entries(
                    **{k: v for k, v in blog_filters.items() if k != "sort"}, selected=False
                )
                selected_matching_blogs = db.count_blog_entries(
                    **{k: v for k, v in blog_filters.items() if k != "sort"}, selected=True
                )
                if blog_total:
                    blog_rows = db.list_blog_entries(
                        **blog_filters,
                        selected=False,
                        limit=(filters.page_size if content_filter == "blogs" else prefetch),
                        offset=(offset if content_filter == "blogs" else 0),
                        table_view=True,
                    )

            selected_paper_rows, selected_paper_total = db.search_papers(
                selected=True,
                limit=SELECTED_CONTENT_PREVIEW_LIMIT,
                offset=0,
                sort="discovered",
                table_view=True,
            )
            selected_blog_total = (
                db.count_blog_entries(selected=True) if has_blog_entries else 0
            )
            selected_blog_rows = (
                db.list_blog_entries(
                    limit=SELECTED_CONTENT_PREVIEW_LIMIT,
                    selected=True,
                    sort="discovered",
                    table_view=True,
                )
                if selected_blog_total
                else []
            )
            selected_state = db.selected_content_state()
            author_rows = db.list_authors_with_tags(tracked_only=True)
            tag_rows = db.list_tags()
            content_tag_rows = db.list_content_tags()
            topic_rows = db.list_topics()
            saved_queries = [_query_view(row, config) for row in db.list_saved_queries()]

            paper_views = [_paper_table_view(db, row) for row in paper_rows]
            selected_paper_views = [_paper_table_view(db, row) for row in selected_paper_rows]

            def blog_view(row: Any) -> dict[str, Any]:
                entry = dict(row)
                raw_authors = _json_value(entry.get("tracked_authors_json"), [])
                entry["tracked_authors"] = [
                    {
                        "name": str(author.get("name") or "Unknown author"),
                        "url": f"/authors/{quote_plus(str(author.get('openalex_id') or ''))}/blogs",
                    }
                    for author in raw_authors
                    if isinstance(author, dict) and author.get("openalex_id")
                ]
                entry["source_site"], entry["source_site_url"] = _website_view(
                    str(entry.get("source_url") or entry.get("url") or "")
                )
                return entry

            blog_views = [blog_view(row) for row in blog_rows]
            selected_blog_views = [blog_view(row) for row in selected_blog_rows]

        def prepare_record(item: dict[str, Any], content_type: str) -> dict[str, Any]:
            item["content_type"] = content_type
            if content_type == "paper":
                item["record_date"] = item.get("publication_date") or item.get("discovered_at") or ""
                item["record_fetched"] = item.get("fetched_at") or item.get("discovered_at") or ""
            else:
                item["record_date"] = item.get("published_at") or item.get("fetched_at") or ""
                item["record_fetched"] = item.get("fetched_at") or ""
            return item

        records = [prepare_record(item, "paper") for item in paper_views]
        records.extend(prepare_record(item, "blog") for item in blog_views)
        selected_records = [prepare_record(item, "paper") for item in selected_paper_views]
        selected_records.extend(prepare_record(item, "blog") for item in selected_blog_views)
        selected_records.sort(key=lambda item: _record_sort_key(item, "discovered"), reverse=True)
        selected_records = selected_records[:SELECTED_CONTENT_PREVIEW_LIMIT]

        if content_filter == "all":
            reverse = filters.sort not in {"oldest", "title"}
            records.sort(key=lambda item: _record_sort_key(item, filters.sort), reverse=reverse)
            records = records[offset : offset + filters.page_size]

        unselected_total = paper_total + blog_total
        matching_total = unselected_total + selected_matching_papers + selected_matching_blogs
        pages = max(1, (unselected_total + filters.page_size - 1) // filters.page_size)
        if filters.page > pages and unselected_total:
            items = filters.query_items(page=pages)
            if content_filter != "all":
                items.append(("content", content_filter))
            return RedirectResponse(f"/?{urlencode(items)}", status_code=303)

        visible_papers = [item for item in records if item["content_type"] == "paper"]
        visible_blogs = [item for item in records if item["content_type"] == "blog"]
        previous_items = filters.query_items(page=filters.page - 1)
        next_items = filters.query_items(page=filters.page + 1)
        page_jump_items = filters.selection_items()
        if filters.page_size != DEFAULT_PAGE_SIZE:
            page_jump_items.append(("page_size", str(filters.page_size)))
        if content_filter != "all":
            previous_items.append(("content", content_filter))
            next_items.append(("content", content_filter))
            page_jump_items.append(("content", content_filter))
        return render(
            request,
            "content.html",
            records=records,
            selected_records=selected_records,
            selected_count=selected_paper_total + selected_blog_total,
            selected_paper_count=selected_paper_total,
            selected_blog_count=selected_blog_total,
            selected_visible_count=len(selected_records),
            selected_matching_count=selected_matching_papers + selected_matching_blogs,
            selected_all_starred=(
                selected_state["total"] > 0
                and selected_state["starred"] == selected_state["total"]
            ),
            selected_all_read=(
                selected_state["total"] > 0
                and selected_state["read"] == selected_state["total"]
            ),
            papers=visible_papers,
            blog_entries=visible_blogs,
            filters=filters,
            content_filter=content_filter,
            authors=author_rows,
            author_tags=tag_rows,
            content_tags=content_tag_rows,
            topics=topic_rows,
            saved_queries=saved_queries,
            result_queries=saved_queries,
            selected_query_id=selected_query_id,
            model_profiles=config.available_llms,
            total=unselected_total,
            matching_total=matching_total,
            paper_total=paper_total + selected_matching_papers,
            blog_total=blog_total + selected_matching_blogs,
            pages=pages,
            is_home=is_home,
            previous_query=urlencode(previous_items),
            next_query=urlencode(next_items),
            page_jump_items=page_jump_items,
        )

    @app.get("/", response_class=HTMLResponse)
    def recent_papers(request: Request) -> HTMLResponse:
        return _render_content_page(request, default_content="all", is_home=True)

    @app.post("/content/{content_type}/{record_id}/star")
    async def star_content(
        request: Request, content_type: str, record_id: int
    ) -> Response:
        form = await request.form()
        starred = str(form.get("starred", "")) in {"1", "true", "on"}
        with Database(config.paths.database) as db:
            if not db.set_content_starred(content_type, record_id, starred):
                raise HTTPException(404, "Content not found")
        if request.headers.get("X-Paperwatch-Async") == "1":
            return JSONResponse({"ok": True, "starred": starred})
        return RedirectResponse(_request_return_path(request), status_code=303)

    @app.post("/content/{content_type}/{record_id}/read")
    async def read_content(
        request: Request, content_type: str, record_id: int
    ) -> Response:
        form = await request.form()
        read = str(form.get("read", "1")) in {"1", "true", "on"}
        with Database(config.paths.database) as db:
            if not db.set_content_read(content_type, record_id, read):
                raise HTTPException(404, "Content not found")
        if request.headers.get("X-Paperwatch-Async") == "1":
            return JSONResponse({"ok": True, "read": read})
        return RedirectResponse(_request_return_path(request), status_code=303)

    @app.post("/content/{content_type}/{record_id}/select")
    async def select_content(
        request: Request, content_type: str, record_id: int
    ) -> Response:
        form = await request.form()
        selected = str(form.get("selected", "")) in {"1", "true", "on"}
        with Database(config.paths.database) as db:
            if not db.set_content_selected(content_type, record_id, selected):
                raise HTTPException(404, "Content not found")
        if request.headers.get("X-Paperwatch-Async") == "1":
            return JSONResponse({"ok": True, "selected": selected})
        return RedirectResponse(_request_return_path(request), status_code=303)

    @app.post("/content/select-results")
    async def select_content_results(request: Request) -> Response:
        form = await request.form()
        filters = _parse_filter_values(form)
        content_filter = _content_kind(form.get("content"), "all")
        selected = str(form.get("selected", "1")) in {"1", "true", "on"}
        target_state = not selected
        paper_filters = {
            "query": filters.query,
            "title_query": filters.title_query,
            "author_id": filters.author_id,
            "author_ids": filters.author_ids,
            "author_tags": filters.author_tags,
            "content_tags": filters.content_tags,
            "topic_id": filters.topic_id,
            "topic_ids": filters.topic_ids,
            "paper_kind": filters.paper_kind,
            "scope": filters.scope,
            "has_pdf": filters.has_pdf,
            "analyzed": filters.analyzed,
            "starred": filters.starred,
            "read": filters.read,
            "notes": filters.notes,
            "selected": target_state,
            "date_from": filters.date_from,
            "date_to": filters.date_to,
        }
        blog_filters = {
            "query": filters.query,
            "title_query": filters.title_query,
            "tags": filters.content_tags,
            "author_id": filters.author_id,
            "author_ids": filters.author_ids,
            "author_tags": filters.author_tags,
            "date_from": filters.date_from,
            "date_to": filters.date_to,
            "starred": filters.starred,
            "read": filters.read,
            "notes": filters.notes,
            "selected": target_state,
            "analyzed": filters.analyzed,
            "show_stubs": filters.display_stubs,
        }
        blog_compatible = not (
            filters.topic_id
            or filters.topic_ids
            or filters.paper_kind
            or filters.scope
            or filters.has_pdf is True
        )
        with Database(config.paths.database) as db:
            changed = db.bulk_set_filtered_content_selected(
                selected=selected,
                paper_filters=(paper_filters if content_filter != "blogs" else None),
                blog_filters=(
                    blog_filters
                    if content_filter != "papers" and blog_compatible
                    else None
                ),
            )
        if _async_request(request):
            return JSONResponse({"ok": True, "selected": selected, "changed": changed})
        return RedirectResponse(_request_return_path(request), status_code=303)

    @app.post("/content/selected/{action}")
    def update_selected_content(request: Request, action: str) -> Response:
        try:
            with Database(config.paths.database) as db:
                changed = db.bulk_update_selected_content(action)
        except ValueError as exc:
            if _async_request(request):
                return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
            raise HTTPException(400, str(exc)) from exc
        if _async_request(request):
            return JSONResponse({"ok": True, "changed": changed})
        return RedirectResponse(_request_return_path(request), status_code=303)

    @app.post("/content/selected/tags/{action}")
    async def update_selected_tags(request: Request, action: str) -> Response:
        form = await request.form()
        try:
            with Database(config.paths.database) as db:
                changed, action_id = db.bulk_update_selected_tags(
                    action, str(form.get("tag", ""))
                )
        except ValueError as exc:
            if request.headers.get("X-Paperwatch-Async") == "1":
                return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
            return RedirectResponse(
                _notice_url(_request_return_path(request), str(exc), level="error"),
                status_code=303,
            )
        payload = {"ok": True, "changed": changed, "action_id": action_id}
        if request.headers.get("X-Paperwatch-Async") == "1":
            return JSONResponse(payload)
        return RedirectResponse(
            _notice_url(_request_return_path(request), f"Updated {changed} tag link(s)."),
            status_code=303,
        )

    @app.post("/tags/remove-global")
    async def remove_global_tag(request: Request) -> Response:
        form = await request.form()
        try:
            with Database(config.paths.database) as db:
                changed, action_id = db.remove_tag_globally(
                    str(form.get("tag", "")), str(form.get("scope", "all"))
                )
        except ValueError as exc:
            if request.headers.get("X-Paperwatch-Async") == "1":
                return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
            return RedirectResponse(
                _notice_url(_request_return_path(request), str(exc), level="error"),
                status_code=303,
            )
        if request.headers.get("X-Paperwatch-Async") == "1":
            return JSONResponse({"ok": True, "changed": changed, "action_id": action_id})
        return RedirectResponse(
            _notice_url(_request_return_path(request), f"Removed {changed} tag link(s)."),
            status_code=303,
        )

    @app.post("/tags/undo")
    async def undo_tag_action(request: Request) -> Response:
        with Database(config.paths.database) as db:
            changed = db.undo_last_tag_action()
        if request.headers.get("X-Paperwatch-Async") == "1":
            return JSONResponse({"ok": True, "changed": changed})
        return RedirectResponse(
            _notice_url(_request_return_path(request), f"Restored {changed} tag link(s)."),
            status_code=303,
        )

    @app.post("/notes/{object_type}/{object_id}")
    async def save_object_notes(
        request: Request, object_type: str, object_id: int
    ) -> Response:
        form = await request.form()
        try:
            with Database(config.paths.database) as db:
                if not db.set_object_notes(object_type, object_id, str(form.get("notes", ""))):
                    raise HTTPException(404, "Object not found")
        except ValueError as exc:
            if request.headers.get("X-Paperwatch-Async") == "1":
                return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
            return RedirectResponse(
                _notice_url(_request_return_path(request), str(exc), level="error"),
                status_code=303,
            )
        if request.headers.get("X-Paperwatch-Async") == "1":
            return JSONResponse({"ok": True})
        return RedirectResponse(_request_return_path(request), status_code=303)

    @app.get("/exports/authors.csv")
    def export_authors_csv() -> Response:
        output = io.StringIO(newline="")
        fields = ["Author Name", "Institution", "OpenAlex ID", "Starred", "Enabled", "Tags", "Notes"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        with Database(config.paths.database) as db:
            for row in db.list_authors_with_tags(tracked_only=True):
                writer.writerow({
                    "Author Name": row["name"],
                    "Institution": row["institution"] or "",
                    "OpenAlex ID": "" if str(row["openalex_id"]).startswith("LOCAL-") else row["openalex_id"],
                    "Starred": "yes" if row["starred"] else "no",
                    "Enabled": "yes" if row["enabled"] else "no",
                    "Tags": row["tags"] or "",
                    "Notes": row["notes"] or "",
                })
        return Response(output.getvalue(), media_type="text/csv; charset=utf-8", headers={"Content-Disposition": 'attachment; filename="paperwatch-authors.csv"'})

    @app.get("/exports/blogs.csv")
    def export_blogs_csv() -> Response:
        output = io.StringIO(newline="")
        fields = [
            "Researcher / author", "Current lab / organization",
            "Primary applied-ML source", "Primary source URL", "Relationship to author",
            "Primary source type", "Primary source mode", "Primary content selector",
            "Primary article URL pattern", "Primary tags", "Notes", "Enabled", "Dead",
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        with Database(config.paths.database) as db:
            for author in db.list_authors(tracked_only=True):
                for blog in db.author_blogs(str(author["openalex_id"])):
                    writer.writerow({
                        "Researcher / author": author["name"],
                        "Current lab / organization": author["institution"] or "",
                        "Primary applied-ML source": blog["label"] or blog["title"] or "",
                        "Primary source URL": blog["url"],
                        "Relationship to author": blog["relationship"] or "",
                        "Primary source type": blog["item_stream_mode"] or "",
                        "Primary source mode": blog["source_mode"] or "auto",
                        "Primary content selector": blog["content_selector"] or "",
                        "Primary article URL pattern": blog["article_url_pattern"] or "",
                        "Primary tags": blog["source_tags"] or "",
                        "Notes": blog["notes"] or "",
                        "Enabled": "yes" if blog["enabled"] else "no",
                        "Dead": "yes" if blog["dead"] else "no",
                    })
        return Response(output.getvalue(), media_type="text/csv; charset=utf-8", headers={"Content-Disposition": 'attachment; filename="paperwatch-blogs.csv"'})

    @app.get("/authors", response_class=HTMLResponse)
    def authors(request: Request) -> HTMLResponse:
        raw_tags = request.query_params.getlist("tag")
        active_tags: list[str] = []
        seen_tags: set[str] = set()
        for value in raw_tags:
            tag = " ".join(str(value).strip().split())[:40]
            normalized = tag.casefold()
            if tag and normalized not in seen_tags:
                active_tags.append(tag)
                seen_tags.add(normalized)
            if len(active_tags) >= 50:
                break
        query = request.query_params.get("q", "").strip()[:200]
        starred = _parse_tristate(request.query_params.get("starred"))
        notes = _parse_tristate(request.query_params.get("notes"))
        enabled = _parse_tristate(request.query_params.get("enabled"))
        with Database(config.paths.database) as db:
            rows = [dict(row) for row in db.list_authors_with_tags(
                tracked_only=True,
                tags=active_tags,
                query=query,
                starred=starred,
                notes=notes,
                enabled=enabled,
            )]
            for row in rows:
                row["is_local"] = str(row["openalex_id"]).startswith("LOCAL-")
            tags = db.list_tags()
        return render(
            request,
            "authors.html",
            authors=rows,
            tags=tags,
            active_tags=tuple(active_tags),
            author_query=query,
            author_starred=starred,
            author_notes=notes,
            author_enabled=enabled,
        )

    @app.get("/data", response_class=HTMLResponse)
    def data_management(request: Request) -> HTMLResponse:
        with Database(config.paths.database) as db:
            authors = [dict(row) for row in db.list_authors_with_tags(tracked_only=True)]
        return render(request, "data_management.html", authors=authors)

    @app.post("/authors/manual")
    async def add_manual_author(request: Request) -> RedirectResponse:
        form = await request.form()
        name = clean_text(str(form.get("name", ""))) or ""
        institution = clean_text(str(form.get("institution", "")))
        try:
            if not name:
                raise ValueError("Author name is required")
            openalex_id = _manual_openalex_author_id(str(form.get("openalex_id", "")))
            with Database(config.paths.database) as db:
                if openalex_id:
                    author_db_id = db.add_author(
                        AuthorCandidate(
                            openalex_id=openalex_id,
                            name=name,
                            institution=institution,
                            metadata={"manual": True},
                        )
                    )
                    identifier = openalex_id
                else:
                    author_db_id = db.add_manual_author(name, institution)
                    author_row = db.find_authors_by_name(name)[0]
                    identifier = str(author_row["openalex_id"])
                for tag in _split_tags(str(form.get("tags", ""))):
                    db.add_author_tag(identifier, tag)
                db.set_object_notes("author", author_db_id, str(form.get("notes", "")))
        except ValueError as exc:
            return RedirectResponse(
                _notice_url("/data", str(exc), level="error"), status_code=303
            )
        suffix = " Papers are paused until an OpenAlex ID is provided." if not openalex_id else ""
        return RedirectResponse(
            _notice_url("/data", f"Added {name}.{suffix}"), status_code=303
        )

    @app.post("/authors/blogs/manual")
    async def import_single_author_blog(request: Request) -> RedirectResponse:
        form = await request.form()
        openalex_id = str(form.get("openalex_id", "")).strip()
        try:
            if not openalex_id:
                raise ValueError("Choose an author")
            url = normalize_blog_url(str(form.get("url", "")))
            with Database(config.paths.database) as db:
                blog_id = db.add_author_blog(openalex_id, url)
                for tag in _split_tags(str(form.get("tags", ""))):
                    db.add_blog_source_tag(blog_id, tag)
        except ValueError as exc:
            return RedirectResponse(
                _notice_url("/data", str(exc), level="error"), status_code=303
            )
        return RedirectResponse(
            _notice_url("/data", "Blog linked."), status_code=303
        )

    @app.post("/authors/blogs/import")
    async def import_author_blogs(request: Request) -> RedirectResponse:
        form = await request.form()
        upload = form.get("csv_file")
        create_missing = str(form.get("create_missing", "")) in {"1", "true", "on"}
        import_tags = _split_tags(str(form.get("import_tags", "")))
        if not isinstance(upload, UploadFile) or not upload.filename:
            return RedirectResponse(
                _notice_url("/data", "Choose a CSV file.", level="error"),
                status_code=303,
            )
        try:
            raw = await upload.read(5_000_001)
            if len(raw) > 5_000_000:
                raise ValueError("CSV file exceeds 5 MB")
            text = raw.decode("utf-8-sig")
            with Database(config.paths.database) as db:
                stats = import_blog_links(
                    db,
                    io.StringIO(text, newline=""),
                    create_missing=create_missing,
                    import_tags=import_tags,
                )
        except (UnicodeDecodeError, ValueError) as exc:
            return RedirectResponse(
                _notice_url("/data", f"Could not import blogs: {exc}", level="error"),
                status_code=303,
            )
        finally:
            await upload.close()
        message = (
            f"Imported {stats.links} blog links; created {stats.authors_created} authors, "
            f"skipped {stats.authors_missing} missing and {stats.authors_ambiguous} ambiguous authors."
        )
        return RedirectResponse(_notice_url("/data", message), status_code=303)

    @app.post("/authors/import")
    async def import_authors_csv(request: Request) -> RedirectResponse:
        form = await request.form()
        upload = form.get("csv_file")
        if not isinstance(upload, UploadFile) or not upload.filename:
            return RedirectResponse(
                _notice_url("/data", "Choose an author CSV file.", level="error"),
                status_code=303,
            )
        imported = 0
        import_tags = _split_tags(str(form.get("import_tags", "")))
        try:
            raw = await upload.read(5_000_001)
            if len(raw) > 5_000_000:
                raise ValueError("CSV file exceeds 5 MB")
            stream = io.StringIO(raw.decode("utf-8-sig"), newline="")
            reader = csv.DictReader(stream, skipinitialspace=True)
            if reader.fieldnames:
                reader.fieldnames = [(name or "").strip() for name in reader.fieldnames]
            fields = {str(name).casefold(): str(name) for name in (reader.fieldnames or ())}
            name_column = next(
                (
                    fields[key]
                    for key in ("author name", "name", "researcher / author")
                    if key in fields
                ),
                None,
            )
            if name_column is None:
                raise ValueError("CSV must contain an Author Name column")
            institution_column = next(
                (fields[key] for key in ("institution", "current lab / organization") if key in fields),
                None,
            )
            openalex_column = next(
                (fields[key] for key in ("openalex id", "openalex author id") if key in fields),
                None,
            )
            starred_column = fields.get("starred")
            enabled_column = fields.get("enabled")
            tags_column = fields.get("tags")
            notes_column = next((fields[key] for key in ("notes", "comments") if key in fields), None)
            with Database(config.paths.database) as db:
                for raw_row in reader:
                    row = {str(key).strip(): str(value or "").strip() for key, value in raw_row.items()}
                    name = clean_text(row.get(name_column)) or ""
                    if not name:
                        continue
                    institution = clean_text(row.get(institution_column)) if institution_column else None
                    openalex_id = _manual_openalex_author_id(
                        row.get(openalex_column, "") if openalex_column else ""
                    )
                    if openalex_id:
                        db.add_author(
                            AuthorCandidate(
                                openalex_id=openalex_id,
                                name=name,
                                institution=institution,
                                metadata={"csv_import": True},
                            )
                        )
                        identifier = openalex_id
                    else:
                        db.add_manual_author(name, institution)
                        matches = db.find_authors_by_name(name)
                        identifier = str(matches[0]["openalex_id"]) if matches else ""
                    if identifier and starred_column:
                        db.set_author_starred(
                            identifier,
                            row.get(starred_column, "").casefold() in {"1", "true", "yes", "on", "starred"},
                        )
                    if identifier and enabled_column:
                        db.set_author_enabled(
                            identifier,
                            row.get(enabled_column, "yes").casefold() not in {"0", "false", "no", "off", "disabled"},
                        )
                    if identifier:
                        for tag in (*import_tags, *_split_tags(row.get(tags_column, "") if tags_column else "")):
                            db.add_author_tag(identifier, tag)
                    if identifier and notes_column:
                        stored_author = db.get_author(identifier)
                        if stored_author is not None:
                            db.set_object_notes(
                                "author", int(stored_author["id"]), row.get(notes_column, "")
                            )
                    imported += 1
        except (UnicodeDecodeError, ValueError) as exc:
            return RedirectResponse(
                _notice_url("/data", f"Could not import authors: {exc}", level="error"),
                status_code=303,
            )
        finally:
            await upload.close()
        return RedirectResponse(
            _notice_url("/data", f"Imported {imported} author row{'s' if imported != 1 else ''}."),
            status_code=303,
        )

    @app.get("/authors/{openalex_id}/blogs", response_class=HTMLResponse)
    def author_blogs(request: Request, openalex_id: str) -> HTMLResponse:
        tag = request.query_params.get("tag", "").strip()[:80]
        with Database(config.paths.database) as db:
            author = db.get_author(openalex_id)
            if author is None or not author["tracked"]:
                raise HTTPException(404, "Author not found")
            blogs = db.author_blogs(openalex_id, tag=tag)
            content_tags = db.list_content_tags()
        return render(
            request,
            "author_blogs.html",
            author=author,
            blogs=blogs,
            active_tag=tag,
            content_tags=content_tags,
            source_modes=sorted(SOURCE_MODES),
        )

    @app.post("/authors/{openalex_id}/blogs")
    async def add_author_blog(request: Request, openalex_id: str) -> RedirectResponse:
        form = await request.form()
        try:
            url = normalize_blog_url(str(form.get("url", "")))
            label = str(form.get("label", ""))
            relationship = str(form.get("relationship", ""))
            source_mode = normalize_source_mode(form.get("source_mode", "auto"))
            content_selector = validate_content_selector(form.get("content_selector"))
            article_url_pattern = validate_article_url_pattern(
                form.get("article_url_pattern")
            )
            with Database(config.paths.database) as db:
                blog_id = db.add_author_blog(
                    openalex_id,
                    url,
                    label=label,
                    relationship=relationship,
                    source_mode=source_mode,
                    content_selector=content_selector,
                    article_url_pattern=article_url_pattern,
                )
                for tag in _split_tags(str(form.get("tags", ""))):
                    db.add_blog_source_tag(blog_id, tag)
        except ValueError as exc:
            return RedirectResponse(
                _notice_url(
                    f"/authors/{quote_plus(openalex_id)}/blogs",
                    str(exc),
                    level="error",
                ),
                status_code=303,
            )
        return RedirectResponse(
            _notice_url(f"/authors/{quote_plus(openalex_id)}/blogs", "Blog linked."),
            status_code=303,
        )

    @app.post("/authors/{openalex_id}/blogs/{blog_id}/tags/add")
    async def add_blog_source_tag(
        request: Request, openalex_id: str, blog_id: int
    ) -> Response:
        form = await request.form()
        try:
            with Database(config.paths.database) as db:
                if not db.add_blog_source_tag(blog_id, str(form.get("tag", ""))):
                    raise HTTPException(404, "Blog source not found")
        except ValueError as exc:
            if request.headers.get("X-Paperwatch-Async") == "1":
                return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
            return RedirectResponse(
                _notice_url(f"/authors/{quote_plus(openalex_id)}/blogs", str(exc), level="error"),
                status_code=303,
            )
        if request.headers.get("X-Paperwatch-Async") == "1":
            return JSONResponse({"ok": True})
        return RedirectResponse(f"/authors/{quote_plus(openalex_id)}/blogs", status_code=303)

    @app.post("/authors/{openalex_id}/blogs/{blog_id}/tags/remove")
    async def remove_blog_source_tag(
        request: Request, openalex_id: str, blog_id: int
    ) -> Response:
        form = await request.form()
        with Database(config.paths.database) as db:
            db.remove_blog_source_tag(blog_id, str(form.get("tag", "")))
        if request.headers.get("X-Paperwatch-Async") == "1":
            return JSONResponse({"ok": True})
        return RedirectResponse(f"/authors/{quote_plus(openalex_id)}/blogs", status_code=303)

    @app.post("/authors/{openalex_id}/blogs/{blog_id}/remove")
    def remove_author_blog(openalex_id: str, blog_id: int) -> RedirectResponse:
        with Database(config.paths.database) as db:
            if not db.remove_author_blog(openalex_id, blog_id):
                raise HTTPException(404, "Blog link not found")
        return RedirectResponse(
            _notice_url(f"/authors/{quote_plus(openalex_id)}/blogs", "Blog unlinked."),
            status_code=303,
        )

    @app.post("/authors/{openalex_id}/blogs/{blog_id}/edit")
    async def edit_author_blog(
        request: Request, openalex_id: str, blog_id: int
    ) -> RedirectResponse:
        form = await request.form()
        try:
            url = normalize_blog_url(str(form.get("url", "")))
            source_mode = normalize_source_mode(form.get("source_mode", "auto"))
            content_selector = validate_content_selector(form.get("content_selector"))
            article_url_pattern = validate_article_url_pattern(
                form.get("article_url_pattern")
            )
            with Database(config.paths.database) as db:
                if not db.update_author_blog_url(
                    openalex_id,
                    blog_id,
                    url,
                    source_mode=source_mode,
                    content_selector=content_selector,
                    article_url_pattern=article_url_pattern,
                ):
                    raise HTTPException(404, "Blog link not found")
        except ValueError as exc:
            return RedirectResponse(
                _notice_url(
                    f"/authors/{quote_plus(openalex_id)}/blogs",
                    str(exc),
                    level="error",
                ),
                status_code=303,
            )
        return RedirectResponse(
            _notice_url(
                f"/authors/{quote_plus(openalex_id)}/blogs",
                "Blog URL updated.",
            ),
            status_code=303,
        )

    @app.post("/authors/{openalex_id}/blogs/{blog_id}/availability")
    async def set_author_blog_availability(
        request: Request, openalex_id: str, blog_id: int
    ) -> RedirectResponse:
        form = await request.form()
        broken = str(form.get("broken", "")) in {"1", "true", "on"}
        with Database(config.paths.database) as db:
            if not db.set_author_blog_broken(openalex_id, blog_id, broken):
                raise HTTPException(404, "Blog link not found")
        message = (
            "Blog marked broken; future checks are paused."
            if broken
            else "Blog restored."
        )
        return RedirectResponse(
            _notice_url(f"/authors/{quote_plus(openalex_id)}/blogs", message),
            status_code=303,
        )

    @app.post("/authors/{openalex_id}/blogs/{blog_id}/delete")
    def delete_author_blog_source(openalex_id: str, blog_id: int) -> RedirectResponse:
        with Database(config.paths.database) as db:
            if not db.mark_author_blog_dead(openalex_id, blog_id):
                raise HTTPException(404, "Blog link not found")
        return RedirectResponse(
            _notice_url(
                f"/authors/{quote_plus(openalex_id)}/blogs",
                "Blog source marked dead; it will not be fetched or re-imported.",
            ),
            status_code=303,
        )

    @app.get("/blogs", response_class=HTMLResponse)
    def blog_feed(request: Request) -> HTMLResponse:
        return _render_content_page(request, default_content="blogs")

    @app.get("/blogs/import", response_class=HTMLResponse)
    def import_blog_entry_page(request: Request) -> HTMLResponse:
        with Database(config.paths.database) as db:
            authors = db.list_authors(tracked_only=True)
        return render(request, "import_blog_entry.html", authors=authors)

    @app.post("/blogs/import")
    async def import_blog_entry(request: Request) -> RedirectResponse:
        # The entry text limit is 5 MB, so allow the form parser to receive a
        # value slightly larger than that before applying the domain-level
        # validation below.  Newer Starlette releases otherwise reject the
        # request with a bare 400 response before we can return the normal
        # validation redirect.
        form = await request.form(**({"max_part_size": 5_100_000} if "max_part_size" in inspect.signature(request.form).parameters else {}))
        author_id = str(form.get("author_id", "")).strip()
        title = clean_text(str(form.get("entry_title", "")))
        content = str(form.get("content", "")).strip()
        published_at = str(form.get("published_at", "")).strip()
        try:
            if not author_id:
                raise ValueError("Choose a tracked author")
            if not title:
                raise ValueError("Entry title is required")
            if not content:
                raise ValueError("Entry text is required")
            if len(title) > 1000:
                raise ValueError("Entry title is too long")
            if len(content) > 5_000_000:
                raise ValueError("Entry text is too long")
            blog_url = normalize_blog_url(str(form.get("blog_url", "")))
            entry_url = normalize_blog_url(str(form.get("entry_url", "")))
            if published_at:
                try:
                    datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                except ValueError as exc:
                    raise ValueError("Published time must be ISO 8601") from exc
            normalized = f"Title: {title}\nURL: {entry_url}\n\n{content}"
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            external_id = hashlib.sha256(
                f"manual\0{entry_url}".encode("utf-8")
            ).hexdigest()
            with Database(config.paths.database) as db:
                if db.get_author(author_id) is None:
                    raise ValueError("Tracked author was not found")
                blog_id = db.add_author_blog(
                    author_id,
                    blog_url,
                    label=str(form.get("blog_label", "")),
                    relationship=str(form.get("relationship", "Direct import")),
                    source_mode="article",
                )
                previous = db.latest_blog_entry(
                    blog_id, entry_kind="snapshot", external_id=external_id
                )
                if previous is not None and previous["content_sha256"] == digest:
                    entry_id = int(previous["id"])
                else:
                    kind, diff = blog_diff(
                        str(previous["normalized_text"]) if previous is not None else None,
                        normalized,
                    )
                    entry_id = db.save_blog_entry(
                        blog_id=blog_id,
                        title=title,
                        content_sha256=digest,
                        normalized_text=normalized,
                        diff_text=diff,
                        change_kind="changed" if previous is None else kind,
                        previous_entry_id=int(previous["id"]) if previous is not None else None,
                        entry_kind="snapshot",
                        external_id=external_id,
                        source_url=entry_url,
                        published_at=published_at or None,
                    )
                for tag in _split_tags(str(form.get("tags", ""))):
                    db.add_blog_entry_tag(entry_id, tag)
                if str(form.get("notes", "")).strip():
                    db.set_object_notes("blog", entry_id, str(form.get("notes", "")))
        except ValueError as exc:
            return RedirectResponse(
                _notice_url("/blogs/import", str(exc), level="error"), status_code=303
            )
        return RedirectResponse(
            _notice_url(f"/blogs/{entry_id}", "Blog entry imported."), status_code=303
        )

    @app.get("/blogs/{entry_id}", response_class=HTMLResponse)
    def blog_entry(request: Request, entry_id: int) -> HTMLResponse:
        with Database(config.paths.database) as db:
            entry = db.get_blog_entry(entry_id)
            prompt_runs = [
                _query_output(row) for row in db.query_runs_for_blog_entry(entry_id)
            ]
            entry_tags = db.blog_entry_tags(entry_id)
            content_tags = db.list_content_tags()
            tracked_authors = db.blog_authors(int(entry["blog_id"])) if entry else []
            classifications = db.classification_results_for_content("blog", entry_id)
        if entry is None:
            raise HTTPException(404, "Blog update not found")
        entry_view = dict(entry)
        entry_view["diff_text"] = compact_blog_diff(str(entry_view.get("diff_text") or ""))
        entry_view["tracked_authors"] = [
            {
                "name": str(author["name"]),
                "url": f"/authors/{quote_plus(str(author['openalex_id']))}/blogs",
                "entries_url": f"/?content=blogs&author={quote_plus(str(author['openalex_id']))}",
            }
            for author in tracked_authors
        ]
        entry_view["display_html"] = str(entry["rendered_html"] or "") or _render_stored_markdown(
            str(entry["normalized_text"] or "")
        )
        return render(
            request,
            "blog_detail.html",
            entry=entry_view,
            prompt_runs=prompt_runs,
            entry_tags=entry_tags,
            content_tags=content_tags,
            classifications=classifications,
        )

    @app.post("/blogs/{entry_id}/delete")
    def delete_blog_source(entry_id: int) -> RedirectResponse:
        with Database(config.paths.database) as db:
            if not db.mark_blog_entry_source_dead(entry_id):
                raise HTTPException(404, "Blog update not found")
        return RedirectResponse(
            _notice_url(
                "/?content=blogs",
                "Blog source marked dead; it will not be fetched or re-imported.",
            ),
            status_code=303,
        )

    @app.post("/blogs/{entry_id}/prompt")
    async def prompt_blog_entry(request: Request, entry_id: int) -> Response:
        async_request = request.headers.get("x-paperwatch-async") == "1"
        form = await request.form()
        query_id = _int_or_none(form.get("query_id"))
        with Database(config.paths.database) as db:
            entry = db.get_blog_entry(entry_id)
            query = db.get_saved_query(query_id) if query_id is not None else None
        if entry is None:
            raise HTTPException(404, "Blog update not found")

        def failure(message: str, status_code: int = 400) -> Response:
            if async_request:
                return JSONResponse({"ok": False, "message": message}, status_code=status_code)
            return RedirectResponse(
                _notice_url(f"/blogs/{entry_id}", message, level="error"), status_code=303
            )

        if query is None:
            return failure("Select a saved prompt.")
        try:
            query_view, profiles, context, source_scope, pdf_strategy = (
                _saved_query_run_options(
                    form, query, config, content_type="blogs"
                )
            )
        except ValueError as exc:
            return failure(str(exc))
        try:
            batch_id = collection_controller.start(
                int(query["id"]),
                profiles,
                context,
                selection={
                    "content_type": "blogs",
                    "blog_entry_ids": [entry_id],
                    "labels": [f"Blog entry: {entry['title']}"],
                },
                source_scope=source_scope,
                pdf_strategy=pdf_strategy,
                content_type="blogs",
            )
        except ValueError as exc:
            return failure(str(exc))
        if batch_id is None:
            return failure("Another saved prompt is already running.", status_code=409)
        runner = "local regexp" if query_view["local_regexp"] else f"{len(set(profiles))} model(s)"
        message = f"Started {query['name']} with {runner}."
        if async_request:
            return JSONResponse({"ok": True, "batch_id": batch_id, "message": message})
        return RedirectResponse(_notice_url(f"/blogs/{entry_id}", message), status_code=303)

    @app.post("/blogs/{entry_id}/tags/add")
    async def add_blog_content_tag(request: Request, entry_id: int) -> Response:
        form = await request.form()
        tag = str(form.get("tag", ""))
        try:
            with Database(config.paths.database) as db:
                if not db.add_blog_entry_tag(entry_id, tag):
                    raise HTTPException(404, "Blog entry not found")
        except ValueError as exc:
            if _async_request(request):
                return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
            return RedirectResponse(_notice_url(f"/blogs/{entry_id}", str(exc), level="error"), status_code=303)
        if _async_request(request):
            return JSONResponse({"ok": True, "tag": tag})
        return RedirectResponse(_notice_url(f"/blogs/{entry_id}", "Tag added."), status_code=303)

    @app.post("/blogs/{entry_id}/tags/remove")
    async def remove_blog_content_tag(request: Request, entry_id: int) -> Response:
        form = await request.form()
        tag = str(form.get("tag", ""))
        try:
            with Database(config.paths.database) as db:
                db.remove_blog_entry_tag(entry_id, tag)
        except ValueError as exc:
            if _async_request(request):
                return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
            return RedirectResponse(_notice_url(f"/blogs/{entry_id}", str(exc), level="error"), status_code=303)
        if _async_request(request):
            return JSONResponse({"ok": True, "tag": tag})
        return RedirectResponse(f"/blogs/{entry_id}", status_code=303)

    @app.post("/content/{content_type}/{record_id}/classifications/{query_id}")
    async def update_content_classification(
        request: Request,
        content_type: str,
        record_id: int,
        query_id: int,
    ) -> Response:
        form = await request.form()
        value = str(form.get("value", ""))
        try:
            with Database(config.paths.database) as db:
                if not db.set_content_classification(query_id, content_type, record_id, value):
                    raise HTTPException(404, "Classification or content not found")
        except ValueError as exc:
            if _async_request(request):
                return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
            return RedirectResponse(_notice_url(_request_return_path(request), str(exc), level="error"), status_code=303)
        if _async_request(request):
            return JSONResponse({"ok": True, "value": value})
        return RedirectResponse(_request_return_path(request), status_code=303)

    @app.post("/authors/search")
    async def search_authors(request: Request) -> RedirectResponse:
        form = await request.form()
        name = str(form.get("name", "")).strip()[:200]
        target = f"/discover?mode=author&q={quote_plus(name)}" if name else "/discover?mode=author"
        return RedirectResponse(target, status_code=303)

    @app.post("/authors/{openalex_id}/add")
    def add_author(openalex_id: str) -> RedirectResponse:
        client = client_factory(config)
        try:
            candidate = client.get_author(openalex_id)
        except Exception as exc:
            return RedirectResponse(
                _notice_url("/authors", f"Could not add author: {exc}", level="error"),
                status_code=303,
            )
        finally:
            client.close()
        with Database(config.paths.database) as db:
            db.add_author(candidate)
        return RedirectResponse(
            _notice_url("/authors", f"Added {candidate.name}."), status_code=303
        )

    @app.post("/authors/{openalex_id}/openalex-id")
    async def update_author_openalex_id(
        request: Request, openalex_id: str
    ) -> RedirectResponse:
        form = await request.form()
        try:
            replacement = _manual_openalex_author_id(
                str(form.get("openalex_id", ""))
            )
            if replacement is None:
                raise ValueError("OpenAlex author ID is required")
            with Database(config.paths.database) as db:
                if not db.update_author_openalex_id(openalex_id, replacement):
                    raise HTTPException(404, "Author not found")
        except ValueError as exc:
            return RedirectResponse(
                _notice_url("/authors", str(exc), level="error"), status_code=303
            )
        return RedirectResponse(
            _notice_url("/authors", f"OpenAlex ID updated to {replacement}."),
            status_code=303,
        )

    @app.post("/authors/{openalex_id}/enable")
    def enable_author(openalex_id: str) -> RedirectResponse:
        with Database(config.paths.database) as db:
            if not db.set_author_enabled(openalex_id, True):
                raise HTTPException(404, "Author not found")
        return RedirectResponse(_notice_url("/authors", "Author enabled."), status_code=303)

    @app.post("/authors/{openalex_id}/star")
    async def star_author(request: Request, openalex_id: str) -> Response:
        form = await request.form()
        starred = str(form.get("starred", "")) in {"1", "true", "on"}
        with Database(config.paths.database) as db:
            if not db.set_author_starred(openalex_id, starred):
                raise HTTPException(404, "Author not found")
        if request.headers.get("X-Paperwatch-Async") == "1":
            return JSONResponse({"ok": True, "starred": starred})
        return RedirectResponse(_request_return_path(request, "/authors"), status_code=303)

    @app.post("/authors/{openalex_id}/disable")
    def disable_author(openalex_id: str) -> RedirectResponse:
        with Database(config.paths.database) as db:
            if not db.set_author_enabled(openalex_id, False):
                raise HTTPException(404, "Author not found")
        return RedirectResponse(_notice_url("/authors", "Author disabled."), status_code=303)

    @app.post("/authors/{openalex_id}/remove")
    def remove_author(openalex_id: str) -> RedirectResponse:
        with Database(config.paths.database) as db:
            if not db.remove_author(openalex_id):
                raise HTTPException(404, "Author not found")
        return RedirectResponse(_notice_url("/authors", "Author removed from tracking; its database record was retained."), status_code=303)

    @app.post("/authors/{openalex_id}/tags")
    async def add_author_tag(request: Request, openalex_id: str) -> Response:
        form = await request.form()
        tag = str(form.get("tag", "")).strip()
        try:
            with Database(config.paths.database) as db:
                if not db.add_author_tag(openalex_id, tag):
                    raise HTTPException(404, "Author not found")
        except ValueError as exc:
            if _async_request(request):
                return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
            return RedirectResponse(_notice_url("/authors", str(exc), level="error"), status_code=303)
        if _async_request(request):
            return JSONResponse({"ok": True, "tag": tag})
        return RedirectResponse(_notice_url("/authors", f"Added tag {tag}."), status_code=303)

    @app.post("/authors/{openalex_id}/tags/remove")
    async def remove_author_tag(request: Request, openalex_id: str) -> Response:
        form = await request.form()
        tag = str(form.get("tag", "")).strip()
        with Database(config.paths.database) as db:
            db.remove_author_tag(openalex_id, tag)
        if _async_request(request):
            return JSONResponse({"ok": True, "tag": tag})
        return RedirectResponse(_notice_url("/authors", f"Removed tag {tag}."), status_code=303)

    def load_remote_paper(reference: str) -> PaperRecord:
        value = reference.strip()
        if not value:
            raise ValueError("Remote paper reference is required")
        arxiv_id = extract_arxiv_id(value)
        if arxiv_id:
            arxiv_client = arxiv_factory(config)
            try:
                entry = arxiv_client.by_id(arxiv_id)
            finally:
                arxiv_client.close()
            if entry is None:
                raise ValueError(f"arXiv paper {arxiv_id} was not found")
            return _paper_from_arxiv(entry)
        openalex_id = short_openalex_id(value)
        client = client_factory(config)
        try:
            if openalex_id and openalex_id.startswith("W"):
                return client.get_work(openalex_id)
            doi = strip_doi(value)
            if doi and re.fullmatch(r"10\.\d{4,9}/\S+", doi, re.I):
                return client.get_work_by_doi(doi)
        finally:
            client.close()
        raise ValueError("Use an arXiv link/ID, DOI, or OpenAlex work ID")

    @app.get("/papers/import", response_class=HTMLResponse)
    def import_paper_form(request: Request) -> HTMLResponse:
        search_query = request.query_params.get("q", "").strip()[:500]
        selected_reference = request.query_params.get("source", "").strip()[:500]
        paper_results: list[PaperRecord] = []
        selected_paper: PaperRecord | None = None
        import_error = ""
        if selected_reference:
            try:
                selected_paper = load_remote_paper(selected_reference)
            except Exception as exc:
                import_error = str(exc)
        elif search_query:
            try:
                if extract_arxiv_id(search_query):
                    selected_paper = load_remote_paper(search_query)
                    selected_reference = search_query
                else:
                    openalex_id = short_openalex_id(search_query)
                    doi = strip_doi(search_query)
                    if (openalex_id and openalex_id.startswith("W")) or (
                        doi and re.fullmatch(r"10\.\d{4,9}/\S+", doi, re.I)
                    ):
                        selected_paper = load_remote_paper(search_query)
                        selected_reference = search_query
                    else:
                        client = client_factory(config)
                        try:
                            paper_results = client.search_works(search_query, limit=25)
                        finally:
                            client.close()
            except Exception as exc:
                import_error = str(exc)
        return render(
            request,
            "import_paper.html",
            import_values=_paper_import_values(selected_paper) if selected_paper else {},
            import_error=import_error,
            search_query=search_query,
            paper_results=paper_results,
            selected_paper=selected_paper,
            remote_reference=selected_reference,
        )

    @app.post("/papers/import", response_class=HTMLResponse)
    async def import_paper(request: Request) -> Response:
        values: dict[str, str] = {}
        staged: Path | None = None
        backup: Path | None = None
        destination: Path | None = None
        upload: UploadFile | None = None
        remote_reference = ""
        selected_paper: PaperRecord | None = None
        try:
            form = await request.form(
                max_files=1,
                max_fields=24,
                max_part_size=config.fetch.max_pdf_bytes,
            )
            values = {
                key: str(form.get(key, ""))
                for key in (
                    "title",
                    "authors",
                    "source_link",
                    "abstract",
                    "publication_date",
                    "venue",
                    "doi",
                    "arxiv_id",
                    "tags",
                    "notes",
                )
            }
            remote_reference = str(form.get("remote_reference", "")).strip()[:500]
            form_upload = form.get("pdf")
            if isinstance(form_upload, UploadFile) and form_upload.filename:
                upload = form_upload
                paper = _manual_paper_record(values, filename=upload.filename)
                staged = await _stage_uploaded_pdf(
                    upload,
                    directory=config.paths.pdf_dir,
                    max_bytes=config.fetch.max_pdf_bytes,
                )
            else:
                if not remote_reference:
                    source_link = values.get("source_link", "")
                    source_doi = strip_doi(source_link)
                    if extract_arxiv_id(source_link) or (
                        source_doi
                        and re.fullmatch(r"10\.\d{4,9}/\S+", source_doi, re.I)
                    ):
                        remote_reference = source_link
                    else:
                        raise ValueError("PDF file is required")
                paper = selected_paper = load_remote_paper(remote_reference)
                if not paper.pdf_url:
                    raise ValueError("The selected paper has no downloadable PDF")
                paper.metadata = {
                    **paper.metadata,
                    "manual_import": True,
                    "remote_reference": remote_reference,
                    "imported_at": _utcnow(),
                }
                staged = config.paths.pdf_dir / f".remote-import-{uuid.uuid4().hex}.pdf"
                download_pdf(
                    paper.pdf_url,
                    staged,
                    max_bytes=config.fetch.max_pdf_bytes,
                )
            with Database(config.paths.database) as db:
                paper_id, _ = db.upsert_paper(paper, auto_analyze=True)
                stored = db.get_paper(paper_id)
                if stored is None:
                    raise RuntimeError("Imported paper could not be read back")
                destination = config.paths.pdf_dir / safe_pdf_name(
                    str(stored["identity_key"])
                )
            if destination.exists():
                backup = destination.with_suffix(
                    destination.suffix + f".backup-{uuid.uuid4().hex}"
                )
                destination.replace(backup)
            staged.replace(destination)
            staged = None
            with Database(config.paths.database) as db:
                db.set_local_pdf(paper_id, str(destination), pdf_sha256(destination))
                for tag in _split_tags(values.get("tags", "")):
                    db.add_paper_tag(paper_id, tag)
                if values.get("notes", "").strip():
                    db.set_object_notes("paper", paper_id, values["notes"])
            if backup is not None:
                backup.unlink(missing_ok=True)
            return RedirectResponse(
                _notice_url(
                    f"/papers/{paper_id}",
                    f"Imported {paper.title}.",
                ),
                status_code=303,
            )
        except Exception as exc:
            if destination is not None:
                if backup is not None and backup.exists():
                    destination.unlink(missing_ok=True)
                    backup.replace(destination)
                elif staged is None:
                    destination.unlink(missing_ok=True)
            response = render(
                request,
                "import_paper.html",
                import_values=(
                    _paper_import_values(selected_paper) if selected_paper else values
                ),
                import_error=str(exc),
                search_query="",
                paper_results=[],
                selected_paper=selected_paper,
                remote_reference=remote_reference,
            )
            response.status_code = 400
            return response
        finally:
            if upload is not None:
                await upload.close()
            if staged is not None:
                staged.unlink(missing_ok=True)
            if backup is not None:
                backup.unlink(missing_ok=True)

    @app.get("/discover", response_class=HTMLResponse)
    def discover(request: Request) -> HTMLResponse:
        requested_mode = request.query_params.get("mode", "").strip().lower()
        legacy_paper_query = request.query_params.get("paper_q", "").strip()[:500]
        legacy_author_query = request.query_params.get("author_q", "").strip()[:200]
        if requested_mode not in {"paper", "author", "blog"}:
            requested_mode = "author" if legacy_author_query and not legacy_paper_query else "paper"
        query_limit = 2000 if requested_mode == "blog" else 500 if requested_mode == "paper" else 200
        query = request.query_params.get("q", "").strip()[:query_limit]
        if not query:
            query = legacy_paper_query if requested_mode == "paper" else legacy_author_query if requested_mode == "author" else ""
        paper_query = query if requested_mode == "paper" else ""
        author_query = query if requested_mode == "author" else ""
        blog_query = query if requested_mode == "blog" else ""
        limit = _int_or_none(request.query_params.get("limit")) or 100
        limit = min(max(limit, 1), 100)
        paper_results: list[dict[str, Any]] = []
        author_results: list[AuthorCandidate] = []
        blog_results: list[dict[str, Any]] = []
        blog_fetch: BlogFetch | None = None
        paper_error = ""
        author_error = ""
        blog_error = ""
        if paper_query or author_query:
            client = client_factory(config)
            try:
                if paper_query:
                    try:
                        arxiv_id = extract_arxiv_id(paper_query)
                        if arxiv_id:
                            arxiv_client = arxiv_factory(config)
                            try:
                                records = _search_arxiv_identifier(
                                    paper_query,
                                    openalex=client,
                                    arxiv=arxiv_client,
                                    limit=limit,
                                )
                            finally:
                                arxiv_client.close()
                        else:
                            records = client.search_works(paper_query, limit=limit)
                        with Database(config.paths.database) as db:
                            paper_results = [
                                _discover_paper_result(db, paper) for paper in records
                            ]
                    except Exception as exc:
                        paper_error = str(exc)
                if author_query:
                    try:
                        author_results = client.search_authors(author_query, limit=limit)
                    except Exception as exc:
                        author_error = str(exc)
            finally:
                client.close()
        if blog_query:
            blog_client = blog_factory(config)
            try:
                try:
                    normalized = normalize_blog_url(blog_query)
                    blog_fetch = _fetch_blog_preview(blog_client, normalized)
                    blog_results = _blog_preview_entries(blog_fetch, normalized)
                    blog_query = normalized
                except Exception as exc:
                    blog_error = str(exc)
            finally:
                close = getattr(blog_client, "close", None)
                if callable(close):
                    close()
        return render(
            request,
            "discover.html",
            discover_mode=requested_mode,
            discover_query=query,
            paper_query=paper_query,
            author_query=author_query,
            blog_query=blog_query,
            search_limit=limit,
            paper_results=paper_results,
            author_results=author_results,
            blog_results=blog_results,
            blog_fetch=blog_fetch,
            paper_error=paper_error,
            author_error=author_error,
            blog_error=blog_error,
        )

    @app.get("/discover/paper", response_class=HTMLResponse)
    def discover_paper_preview(request: Request) -> HTMLResponse:
        work_id = short_openalex_id(request.query_params.get("work_id", ""))
        query = request.query_params.get("q", "").strip()[:500]
        if not work_id or not work_id.startswith("W"):
            raise HTTPException(400, "OpenAlex work ID is required")
        client = client_factory(config)
        try:
            paper = client.get_work(work_id)
        except Exception as exc:
            raise HTTPException(404, f"Could not load paper: {exc}") from exc
        finally:
            client.close()
        with Database(config.paths.database) as db:
            result = _discover_paper_result(db, paper)
        return render(
            request,
            "discover_paper.html",
            result=result,
            paper=paper,
            paper_query=query,
        )

    @app.post("/api/discover/paper-import")
    async def discover_paper_import(request: Request) -> JSONResponse:
        form = await request.form()
        work_id = short_openalex_id(str(form.get("work_id", "")))
        if not work_id or not work_id.startswith("W"):
            return JSONResponse({"error": "OpenAlex work ID is required"}, status_code=400)
        with Database(config.paths.database) as db:
            stored = db.get_paper_by_openalex_id(work_id)
            if stored is not None:
                local_path = str(stored["local_pdf_path"] or "").strip()
                if local_path and Path(local_path).expanduser().is_file():
                    return JSONResponse(
                        {
                            "state": "imported",
                            "paper_id": int(stored["id"]),
                            "url": f"/papers/{int(stored['id'])}",
                        }
                    )
        snapshot = controller.snapshot()
        if snapshot.running:
            same_work = snapshot.mode == "download" and (
                not snapshot.query or snapshot.query == work_id
            )
            if same_work:
                return JSONResponse({"state": "importing"}, status_code=202)
            return JSONResponse(
                {"error": "Another fetch is already running."}, status_code=409
            )
        parameters = inspect.signature(controller.start).parameters
        try:
            if "query" in parameters:
                started = controller.start("download", query=work_id)
            else:
                started = controller.start("download")
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        if not started:
            return JSONResponse(
                {"error": "Another fetch is already running."}, status_code=409
            )
        return JSONResponse({"state": "importing"}, status_code=202)

    @app.get("/api/discover/paper-import-status")
    def discover_paper_import_status(request: Request) -> JSONResponse:
        work_id = short_openalex_id(request.query_params.get("work_id", ""))
        if not work_id or not work_id.startswith("W"):
            return JSONResponse({"error": "OpenAlex work ID is required"}, status_code=400)
        with Database(config.paths.database) as db:
            stored = db.get_paper_by_openalex_id(work_id)
            if stored is not None:
                local_path = str(stored["local_pdf_path"] or "").strip()
                if local_path and Path(local_path).expanduser().is_file():
                    return JSONResponse(
                        {
                            "state": "imported",
                            "paper_id": int(stored["id"]),
                            "url": f"/papers/{int(stored['id'])}",
                        }
                    )
        snapshot = controller.snapshot()
        if snapshot.running and snapshot.mode == "download" and (
            not snapshot.query or snapshot.query == work_id
        ):
            return JSONResponse({"state": "importing"})
        if (
            not snapshot.running
            and snapshot.mode == "download"
            and snapshot.query == work_id
            and snapshot.status == "failed"
        ):
            return JSONResponse(
                {"state": "failed", "error": snapshot.error or snapshot.message or "Import failed"}
            )
        return JSONResponse({"state": "idle"})

    @app.get("/discover/blog/import", response_class=HTMLResponse)
    def discover_blog_import_form(request: Request) -> HTMLResponse:
        source = request.query_params.get("source", "").strip()[:2000]
        entry = request.query_params.get("entry", "").strip()[:2000]
        if not source:
            raise HTTPException(400, "Blog source URL is required")
        try:
            source = normalize_blog_url(source)
            target = normalize_blog_url(entry) if entry else source
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        blog_client = blog_factory(config)
        try:
            try:
                fetched = _fetch_blog_preview(
                    blog_client, target, source_mode="article" if entry else "auto"
                )
                preview_entries = _blog_preview_entries(fetched, target)
                fetch_error = ""
            except Exception as exc:
                fetched = None
                preview_entries = []
                fetch_error = str(exc)
        finally:
            close = getattr(blog_client, "close", None)
            if callable(close):
                close()
        with Database(config.paths.database) as db:
            authors = [dict(row) for row in db.list_authors(tracked_only=True)]
        return render(
            request,
            "discover_blog_import.html",
            source_url=source,
            entry_url=target if entry else "",
            import_scope="entry" if entry else "whole",
            fetched=fetched,
            preview_entries=preview_entries,
            fetch_error=fetch_error,
            authors=authors,
        )

    @app.post("/discover/blog/import")
    async def discover_blog_import(request: Request) -> RedirectResponse:
        form = await request.form()
        source = str(form.get("source", "")).strip()[:2000]
        entry = str(form.get("entry", "")).strip()[:2000]
        scope = str(form.get("scope", "whole")).strip().lower()
        author_openalex_id = str(form.get("author_openalex_id", "")).strip()
        new_author_name = clean_text(str(form.get("new_author_name", ""))) or ""
        label = clean_text(str(form.get("label", ""))) or ""
        relationship = clean_text(str(form.get("relationship", ""))) or ""
        tags = _split_tags(str(form.get("tags", "")))
        notes = str(form.get("notes", ""))
        target = "/discover?mode=blog"
        if source:
            target += f"&q={quote_plus(source)}"
        try:
            normalized_source = normalize_blog_url(source)
            normalized_entry = normalize_blog_url(entry) if entry else ""
            if scope not in {"whole", "entry"}:
                raise ValueError("Unknown blog import scope")
            if scope == "entry" and not normalized_entry:
                raise ValueError("Blog entry URL is required")
            with Database(config.paths.database) as db:
                if new_author_name:
                    db.add_manual_author(new_author_name)
                    matches = db.find_authors_by_name(new_author_name)
                    if not matches:
                        raise ValueError("Could not create author")
                    author_openalex_id = str(matches[0]["openalex_id"])
                elif not author_openalex_id:
                    raise ValueError("Choose an author or enter a new author name")
                else:
                    author = db.get_author(author_openalex_id)
                    if author is None or not author["tracked"]:
                        raise ValueError("Author not found")

            fetch_url = normalized_entry if scope == "entry" else normalized_source
            blog_client = blog_factory(config)
            try:
                fetched = _fetch_blog_preview(
                    blog_client,
                    fetch_url,
                    source_mode="article" if scope == "entry" else "auto",
                )
            finally:
                close = getattr(blog_client, "close", None)
                if callable(close):
                    close()
            with Database(config.paths.database) as db:
                _, saved = _save_discovered_blog_fetch(
                    db,
                    author_openalex_id=author_openalex_id,
                    source_url=fetch_url,
                    fetched=fetched,
                    label=label,
                    relationship=relationship,
                    tags=tags,
                    notes=notes,
                )
        except Exception as exc:
            return RedirectResponse(
                _notice_url(target, f"Could not import blog: {exc}", level="error"),
                status_code=303,
            )
        kind = "page" if scope == "entry" else "blog source"
        return RedirectResponse(
            _notice_url(
                "/blogs",
                f"Imported {kind} and stored {saved} entr{'y' if saved == 1 else 'ies'}.",
            ),
            status_code=303,
        )

    @app.get("/search")
    def search_papers(request: Request) -> RedirectResponse:
        query = request.query_params.get("q", "").strip()[:500]
        target = f"/discover?mode=paper&q={quote_plus(query)}" if query else "/discover?mode=paper"
        return RedirectResponse(target, status_code=303)

    @app.post("/actions/run")
    def start_run(request: Request) -> RedirectResponse:
        if controller.snapshot().running:
            controller.cancel()
        else:
            controller.start("run")
        referer = urlparse(request.headers.get("referer", ""))
        target = referer.path or "/"
        if referer.query:
            target = f"{target}?{referer.query}"
        return RedirectResponse(target, status_code=303)

    @app.post("/actions/download")
    async def download_paper(request: Request) -> RedirectResponse:
        form = await request.form()
        work_id = str(form.get("work_id", "")).strip()[:100]
        title = str(form.get("title", "")).strip()[:500]
        search_query = str(form.get("q", "")).strip()[:500]
        target = (
            f"/discover?mode=paper&q={quote_plus(search_query)}"
            if search_query
            else "/discover?mode=paper"
        )
        if not work_id:
            return RedirectResponse(
                _notice_url(target, "Paper identifier cannot be empty", level="error"),
                status_code=303,
            )
        parameters = inspect.signature(controller.start).parameters
        try:
            if "query" in parameters:
                started = controller.start("download", query=work_id)
            else:
                started = controller.start("download")
        except ValueError as exc:
            return RedirectResponse(
                _notice_url(target, str(exc), level="error"), status_code=303
            )
        if started:
            message = f"Downloading {title or work_id} and fetching linked blogs."
            level = "success"
        else:
            message = "A fetch is already running."
            level = "warning"
        return RedirectResponse(_notice_url(target, message, level=level), status_code=303)

    @app.post("/actions/backfill")
    def start_backfill() -> RedirectResponse:
        if controller.start("backfill"):
            message = (
                "One-year PDF backfill started. Historical papers will not be "
                "auto-summarized."
            )
            level = "success"
        else:
            message = "A fetch is already running."
            level = "warning"
        return RedirectResponse(_notice_url("/discover", message, level=level), status_code=303)

    @app.get("/api/run-status")
    def run_status() -> JSONResponse:
        return JSONResponse(asdict(controller.snapshot()))

    @app.get("/papers", response_class=HTMLResponse)
    def papers(request: Request) -> HTMLResponse:
        return _render_content_page(request, default_content="papers")

    @app.get("/papers/{paper_id}", response_class=HTMLResponse)
    def paper_detail(request: Request, paper_id: int) -> HTMLResponse:
        with Database(config.paths.database) as db:
            paper_row = db.get_paper(paper_id)
            if paper_row is None:
                raise HTTPException(404, "Paper not found")
            paper = _paper_pdf_state(db, _paper_view(paper_row))
            author_rows = [_author_view(row) for row in db.paper_authors(paper_id)]
            default_author_index = next(
                (index for index, author in enumerate(author_rows) if bool(author.get("is_tracked"))),
                0,
            )
            topic_rows = [dict(row) for row in db.paper_topics(paper_id)]
            analysis_tasks = [
                dict(row) for row in db.visible_analysis_tasks_for_paper(paper_id)
            ]
            analyses = [_analysis_view(row) for row in db.paper_analyses(paper_id)]
            prompt_runs = [
                _query_output(row) for row in db.query_runs_for_paper(paper_id)
            ]
            paper_tags = db.paper_tags(paper_id)
            content_tags = db.list_content_tags()
            classifications = db.classification_results_for_content("paper", paper_id)
        return render(
            request,
            "paper_detail.html",
            paper=paper,
            authors=author_rows,
            default_author_index=default_author_index,
            topics=topic_rows,
            analysis_tasks=analysis_tasks,
            analyses=analyses,
            prompt_runs=prompt_runs,
            metadata=_json_value(paper["metadata_json"], {}),
            paper_tags=paper_tags,
            content_tags=content_tags,
            classifications=classifications,
        )

    @app.post("/papers/{paper_id}/delete")
    def delete_paper(paper_id: int) -> RedirectResponse:
        with Database(config.paths.database) as db:
            if not db.mark_paper_dead(paper_id):
                raise HTTPException(404, "Paper not found")
        return RedirectResponse(
            _notice_url(
                "/?content=papers",
                "Paper marked dead; its identifiers and source links will not be fetched again.",
            ),
            status_code=303,
        )

    @app.post("/papers/{paper_id}/authors/{openalex_id}/add")
    def add_paper_author(paper_id: int, openalex_id: str) -> RedirectResponse:
        with Database(config.paths.database) as db:
            if db.get_paper(paper_id) is None:
                raise HTTPException(404, "Paper not found")
        with Database(config.paths.database) as db:
            existing = db.get_author(openalex_id)
            if existing is not None:
                db.set_author_enabled(openalex_id, True)
                name = str(existing["name"])
            else:
                client = client_factory(config)
                try:
                    candidate = client.get_author(openalex_id)
                except Exception as exc:
                    return RedirectResponse(
                        _notice_url(
                            f"/papers/{paper_id}",
                            f"Could not add author: {exc}",
                            level="error",
                        ),
                        status_code=303,
                    )
                finally:
                    client.close()
                db.add_author(candidate)
                name = candidate.name
        return RedirectResponse(
            _notice_url(f"/papers/{paper_id}", f"Added {name} to tracking."),
            status_code=303,
        )

    @app.post("/papers/{paper_id}/tags/add")
    async def add_paper_content_tag(request: Request, paper_id: int) -> Response:
        form = await request.form()
        tag = str(form.get("tag", ""))
        try:
            with Database(config.paths.database) as db:
                if not db.add_paper_tag(paper_id, tag):
                    raise HTTPException(404, "Paper not found")
        except ValueError as exc:
            if _async_request(request):
                return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
            return RedirectResponse(_notice_url(f"/papers/{paper_id}", str(exc), level="error"), status_code=303)
        if _async_request(request):
            return JSONResponse({"ok": True, "tag": tag})
        return RedirectResponse(_notice_url(f"/papers/{paper_id}", "Tag added."), status_code=303)

    @app.post("/papers/{paper_id}/tags/remove")
    async def remove_paper_content_tag(request: Request, paper_id: int) -> Response:
        form = await request.form()
        tag = str(form.get("tag", ""))
        try:
            with Database(config.paths.database) as db:
                db.remove_paper_tag(paper_id, tag)
        except ValueError as exc:
            if _async_request(request):
                return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
            return RedirectResponse(_notice_url(f"/papers/{paper_id}", str(exc), level="error"), status_code=303)
        if _async_request(request):
            return JSONResponse({"ok": True, "tag": tag})
        return RedirectResponse(f"/papers/{paper_id}", status_code=303)

    @app.post("/papers/{paper_id}/analyze")
    async def analyze_paper(request: Request, paper_id: int) -> Response:
        async_request = request.headers.get("x-paperwatch-async") == "1"
        form = await request.form()
        scope = str(form.get("scope", "fulltext"))
        if scope not in SCOPES:
            raise HTTPException(400, "Invalid analysis scope")
        profile_names = tuple(str(value) for value in form.getlist("model"))
        strategy = str(form.get("strategy", "auto"))
        requested_context = _int_or_none(form.get("context_window_tokens"))
        with Database(config.paths.database) as db:
            paper = db.get_paper(paper_id)
        if paper is None:
            raise HTTPException(404, "Paper not found")

        def failure(message: str, status_code: int = 400) -> Response:
            if async_request:
                return JSONResponse({"ok": False, "message": message}, status_code=status_code)
            return RedirectResponse(
                _notice_url(f"/papers/{paper_id}", message, level="error"), status_code=303
            )

        try:
            context_tokens = _configured_context_window(
                config, profile_names, requested_context
            )
        except ValueError as exc:
            return failure(str(exc))
        if scope == "abstract" and not paper["abstract"]:
            return failure("This paper has no abstract to analyze.")
        if scope == "fulltext" and not (paper["pdf_url"] or paper["local_pdf_path"]):
            return failure("This paper has no PDF available for full-text analysis.")
        try:
            if len(__import__("inspect").signature(model_controller.start).parameters) <= 3:
                batch_id = model_controller.start(paper_id, scope, profile_names)
            else:
                batch_id = model_controller.start(
                    paper_id, scope, profile_names, analysis_strategy=strategy,
                    context_window_tokens=context_tokens,
                )
        except ValueError as exc:
            return failure(str(exc))
        if batch_id is None:
            message = "Another model-analysis batch is already running."
            if async_request:
                return JSONResponse({"ok": False, "message": message}, status_code=409)
            return RedirectResponse(
                _notice_url(f"/papers/{paper_id}", message, level="warning"), status_code=303
            )
        message = f"Started {len(set(profile_names))} model analysis task(s)."
        if async_request:
            return JSONResponse({"ok": True, "batch_id": batch_id, "message": message})
        return RedirectResponse(
            _notice_url(f"/papers/{paper_id}#paper-jobs", message), status_code=303
        )

    @app.post("/papers/{paper_id}/prompt")
    async def prompt_paper(request: Request, paper_id: int) -> Response:
        async_request = request.headers.get("x-paperwatch-async") == "1"
        form = await request.form()
        query_id = _int_or_none(form.get("query_id"))
        with Database(config.paths.database) as db:
            paper = db.get_paper(paper_id)
            query = db.get_saved_query(query_id) if query_id is not None else None
        if paper is None:
            raise HTTPException(404, "Paper not found")

        def failure(message: str, status_code: int = 400) -> Response:
            if async_request:
                return JSONResponse({"ok": False, "message": message}, status_code=status_code)
            return RedirectResponse(
                _notice_url(f"/papers/{paper_id}", message, level="error"), status_code=303
            )

        if query is None:
            return failure("Select a saved prompt.")
        try:
            query_view, profiles, context, source_scope, pdf_strategy = (
                _saved_query_run_options(
                    form, query, config, content_type="papers"
                )
            )
        except ValueError as exc:
            return failure(str(exc))
        try:
            batch_id = collection_controller.start(
                int(query["id"]),
                profiles,
                context,
                selection={"paper_ids": [paper_id], "labels": [f"Paper: {paper['title']}"]},
                source_scope=source_scope,
                pdf_strategy=pdf_strategy,
            )
        except ValueError as exc:
            return failure(str(exc))
        if batch_id is None:
            message = "Another saved prompt is already running."
            if async_request:
                return JSONResponse({"ok": False, "message": message}, status_code=409)
            return RedirectResponse(
                _notice_url(f"/papers/{paper_id}", message, level="warning"), status_code=303
            )
        runner = "local regexp" if query_view["local_regexp"] else f"{len(set(profiles))} model(s)"
        message = f"Started {query['name']} with {runner}."
        if async_request:
            return JSONResponse({"ok": True, "batch_id": batch_id, "message": message})
        return RedirectResponse(
            _notice_url(f"/papers/{paper_id}#paper-jobs", message), status_code=303
        )

    @app.get("/api/analysis-status")
    def analysis_status() -> JSONResponse:
        return JSONResponse(asdict(model_controller.snapshot()))

    @app.post("/api/analysis-cancel")
    def cancel_analysis() -> JSONResponse:
        cancelled = model_controller.cancel()
        return JSONResponse(
            {"ok": cancelled, "message": "Cancellation requested" if cancelled else "No analysis is running"},
            status_code=200 if cancelled else 409,
        )

    def local_pdf_path(paper_id: int) -> Path:
        with Database(config.paths.database) as db:
            paper = db.get_paper(paper_id)
        if paper is None or not paper["local_pdf_path"]:
            raise HTTPException(404, "Local PDF not found")
        path = Path(paper["local_pdf_path"]).expanduser().resolve()
        root = config.paths.pdf_dir.resolve()
        if not path.is_file() or (path != root and root not in path.parents):
            raise HTTPException(404, "Local PDF not found")
        return path

    @app.get("/papers/{paper_id}/pdf/open")
    def open_local_pdf(paper_id: int) -> Response:
        path = local_pdf_path(paper_id)
        opener = shutil.which("xdg-open")
        if opener is None:
            raise HTTPException(503, "xdg-open is not available")
        try:
            subprocess.Popen(
                [opener, str(path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            raise HTTPException(500, f"Could not open PDF: {exc}") from exc
        return Response(status_code=204)

    @app.post("/papers/{paper_id}/pdf/upload")
    async def replace_local_pdf(request: Request, paper_id: int) -> RedirectResponse:
        upload: UploadFile | None = None
        staged: Path | None = None
        backup: Path | None = None
        destination: Path | None = None
        try:
            form = await request.form(
                max_files=1,
                max_fields=5,
                max_part_size=config.fetch.max_pdf_bytes,
            )
            candidate = form.get("pdf")
            if not isinstance(candidate, UploadFile) or not candidate.filename:
                raise ValueError("Choose a PDF file")
            upload = candidate
            staged = await _stage_uploaded_pdf(
                upload,
                directory=config.paths.pdf_dir,
                max_bytes=config.fetch.max_pdf_bytes,
            )
            with Database(config.paths.database) as db:
                paper = db.get_paper(paper_id)
                if paper is None:
                    raise HTTPException(404, "Paper not found")
                destination = config.paths.pdf_dir / safe_pdf_name(str(paper["identity_key"]))
            if destination.exists():
                backup = destination.with_suffix(destination.suffix + f".backup-{uuid.uuid4().hex}")
                destination.replace(backup)
            staged.replace(destination)
            staged = None
            with Database(config.paths.database) as db:
                db.set_local_pdf(paper_id, str(destination), pdf_sha256(destination))
            if backup is not None:
                backup.unlink(missing_ok=True)
            return RedirectResponse(
                _notice_url(f"/papers/{paper_id}", "Local PDF replaced."),
                status_code=303,
            )
        except (ValueError, PDFError, OSError) as exc:
            if destination is not None and backup is not None and backup.exists():
                destination.unlink(missing_ok=True)
                backup.replace(destination)
            return RedirectResponse(
                _notice_url(f"/papers/{paper_id}", str(exc), level="error"),
                status_code=303,
            )
        finally:
            if upload is not None:
                await upload.close()
            if staged is not None:
                staged.unlink(missing_ok=True)
            if backup is not None:
                backup.unlink(missing_ok=True)

    @app.post("/papers/{paper_id}/pdf/restore")
    def restore_local_pdf(paper_id: int) -> RedirectResponse:
        with Database(config.paths.database) as db:
            paper = db.get_paper(paper_id)
            if paper is None:
                raise HTTPException(404, "Paper not found")
            pdf_url = str(paper["pdf_url"] or "").strip()
            if not pdf_url:
                return RedirectResponse(
                    _notice_url(
                        f"/papers/{paper_id}",
                        "No remote PDF URL is stored for this paper.",
                        level="error",
                    ),
                    status_code=303,
                )
            destination = config.paths.pdf_dir / safe_pdf_name(str(paper["identity_key"]))
        backup = destination.with_suffix(destination.suffix + f".backup-{uuid.uuid4().hex}")
        try:
            if destination.exists():
                destination.replace(backup)
            download_pdf(
                pdf_url,
                destination,
                max_bytes=config.fetch.max_pdf_bytes,
            )
            validate_pdf_file(destination)
            with Database(config.paths.database) as db:
                db.set_local_pdf(paper_id, str(destination), pdf_sha256(destination))
            backup.unlink(missing_ok=True)
        except (PDFError, OSError) as exc:
            destination.unlink(missing_ok=True)
            if backup.exists():
                backup.replace(destination)
            return RedirectResponse(
                _notice_url(f"/papers/{paper_id}", str(exc), level="error"),
                status_code=303,
            )
        return RedirectResponse(
            _notice_url(f"/papers/{paper_id}", "PDF restored from the remote source."),
            status_code=303,
        )

    @app.get("/papers/{paper_id}/pdf")
    def local_pdf(paper_id: int) -> FileResponse:
        path = local_pdf_path(paper_id)
        return FileResponse(
            path,
            media_type="application/pdf",
            filename=f"paper-{paper_id}.pdf",
            content_disposition_type="inline",
        )

    @app.get("/papers/{paper_id}/export.json")
    def export_paper(paper_id: int) -> JSONResponse:
        with Database(config.paths.database) as db:
            paper = db.get_paper(paper_id, include_dead=True)
            if paper is None:
                raise HTTPException(404, "Paper not found")
            payload = _paper_export(db, paper)
        return JSONResponse(
            payload,
            headers={"Content-Disposition": f'attachment; filename="paper-{paper_id}.json"'},
        )

    @app.get("/analyses/{analysis_id}", response_class=HTMLResponse)
    def analysis_detail(request: Request, analysis_id: int) -> HTMLResponse:
        with Database(config.paths.database) as db:
            analysis_row = db.get_analysis_by_id(analysis_id)
            if analysis_row is None:
                raise HTTPException(404, "Analysis not found")
            paper_row = db.get_paper(int(analysis_row["paper_id"]))
            if paper_row is None:
                raise HTTPException(404, "Paper not found")
        return render(
            request,
            "analysis_detail.html",
            analysis=_analysis_view(analysis_row),
            paper=_paper_view(paper_row),
        )

    @app.get("/analyses/{analysis_id}/export.json")
    def export_analysis(analysis_id: int) -> JSONResponse:
        with Database(config.paths.database) as db:
            analysis = db.get_analysis_by_id(analysis_id)
        if analysis is None:
            raise HTTPException(404, "Analysis not found")
        return JSONResponse(
            _json_value(analysis["result_json"], {}),
            headers={
                "Content-Disposition": f'attachment; filename="analysis-{analysis_id}.json"'
            },
        )

    @app.get("/exports/papers.{format}")
    def export_papers(request: Request, format: str) -> Response:
        if format not in {"json", "jsonl", "csv"}:
            raise HTTPException(404, "Unknown export format")
        filters = _parse_filters(request)
        with Database(config.paths.database) as db:
            records = [
                _paper_export(db, row)
                for row in _all_filtered(db, filters, include_dead=True)
            ]
        if format == "csv":
            return _csv_response(records)
        if format == "jsonl":
            body = "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in records
            )
            return Response(
                body,
                media_type="application/x-ndjson",
                headers={"Content-Disposition": 'attachment; filename="paperwatch-papers.jsonl"'},
            )
        return JSONResponse(
            records,
            headers={"Content-Disposition": 'attachment; filename="paperwatch-papers.json"'},
        )

    @app.get("/queries", response_class=HTMLResponse)
    def queries(request: Request) -> HTMLResponse:
        with Database(config.paths.database) as db:
            saved = [_query_view(row, config) for row in db.list_saved_queries()]
            runs = [_query_output(row) for row in db.list_query_runs(limit=100)]
        runs_by_query: dict[int, list[dict[str, Any]]] = {}
        for row in runs:
            runs_by_query.setdefault(int(row["query_id"]), []).append(row)
        return render(
            request,
            "queries.html",
            queries=saved,
            runs=runs,
            runs_by_query=runs_by_query,
            model_profiles=config.available_llms,
        )

    @app.get("/queries/runs/{run_id}", response_class=HTMLResponse)
    def query_run_detail(request: Request, run_id: int) -> HTMLResponse:
        with Database(config.paths.database) as db:
            row = db.get_query_run(run_id)
            if row is None:
                raise HTTPException(404, "Query output not found")
            run = _query_output(row)
            selection = run.get("selection") if isinstance(run.get("selection"), dict) else {}
            links: list[dict[str, str]] = []
            blog_ids = selection.get("blog_entry_ids")
            if isinstance(blog_ids, list):
                for entry in db.blog_entries_by_ids(blog_ids):
                    links.append(
                        {
                            "label": f"Blog entry: {entry['title']}",
                            "url": f"/blogs/{int(entry['id'])}",
                        }
                    )
            paper_ids = selection.get("paper_ids")
            if isinstance(paper_ids, list):
                for raw_id in paper_ids:
                    paper_id = _int_or_none(raw_id)
                    paper = db.get_paper(paper_id) if paper_id is not None else None
                    if paper is not None:
                        links.append(
                            {
                                "label": f"Paper: {paper['title']}",
                                "url": f"/papers/{paper_id}",
                            }
                        )
            run["selection_links"] = links
            run["selection_source"] = str(
                selection.get("source_scope")
                or ("blog" if selection.get("content_type") == "blogs" else "abstract")
            )
        return render(request, "query_run_detail.html", run=run)

    @app.get("/queries/new", response_class=HTMLResponse)
    def query_new_page(request: Request) -> HTMLResponse:
        selected = [config.available_llms[0].profile] if config.available_llms else []
        query = {
            "name": "",
            "prompt": "",
            "is_builtin": False,
            "model_profiles": selected,
            "paper_source_scope": "auto",
            "pdf_strategy": "chunked",
            "regexp_pattern": "",
            "regexp_radius": 0,
            "regexp_debug_text": "",
            "regexp_mode": "query",
            "content_scope": "both",
            "query_kind": "summary",
            "label_class": "Relevance",
            "label_values": [],
            "label_values_text": (
                r"Relevant = (?i)\b(relevant|yes|include)\b" "\n"
                r"Not relevant = (?i)\b(not relevant|no|exclude)\b"
            ),
            "label_debug_text": "",
            "local_regexp": False,
        }
        return render(
            request,
            "query_edit.html",
            query=query,
            is_new=True,
            model_profiles=config.available_llms,
        )

    @app.get("/queries/{query_id}/edit", response_class=HTMLResponse)
    def query_edit_page(request: Request, query_id: int) -> HTMLResponse:
        with Database(config.paths.database) as db:
            query = db.get_saved_query(query_id)
            if query is None:
                raise HTTPException(404, "Saved query not found")
        return render(
            request,
            "query_edit.html",
            query=_query_view(query, config),
            is_new=False,
            model_profiles=config.available_llms,
        )

    @app.post("/queries")
    async def create_query(request: Request) -> RedirectResponse:
        form = await request.form()
        try:
            execution = _query_execution_form(form, config)
            with Database(config.paths.database) as db:
                query_id = db.create_saved_query(
                    name=str(form.get("name", "")),
                    prompt=str(form.get("prompt", "")),
                    auto_run_papers=str(form.get("auto_run_papers", ""))
                    in {"1", "true", "on"},
                    auto_run_blogs=str(form.get("auto_run_blogs", ""))
                    in {"1", "true", "on"},
                    **execution,
                )
        except ValueError as exc:
            return RedirectResponse(
                _notice_url("/queries/new", str(exc), level="error"), status_code=303
            )
        return RedirectResponse(
            _notice_url(f"/queries/{query_id}/edit", "Saved query created."),
            status_code=303,
        )

    @app.post("/queries/{query_id}/edit")
    async def edit_query(request: Request, query_id: int) -> RedirectResponse:
        form = await request.form()
        try:
            legacy_target_form = not any(
                key in form
                for key in (
                    "model", "paper_source_scope", "pdf_strategy",
                    "content_scope",
                )
            )
            with Database(config.paths.database) as db:
                existing_query = db.get_saved_query(query_id)
                if existing_query is None:
                    raise HTTPException(404, "Saved query not found")
                execution = _query_execution_form(
                    form,
                    config,
                    existing_query=existing_query,
                )
                changed = db.update_saved_query(
                    query_id,
                    name=str(form.get("name", "")),
                    prompt=str(form.get("prompt", "")),
                    auto_run_papers=(
                        str(form.get("auto_run_papers", "")) in {"1", "true", "on"}
                        if legacy_target_form else None
                    ),
                    auto_run_blogs=(
                        str(form.get("auto_run_blogs", "")) in {"1", "true", "on"}
                        if legacy_target_form else None
                    ),
                    **execution,
                )
            if not changed:
                raise HTTPException(404, "Saved query not found")
        except ValueError as exc:
            return RedirectResponse(
                _notice_url(f"/queries/{query_id}/edit", str(exc), level="error"),
                status_code=303,
            )
        return RedirectResponse(
            _notice_url(f"/queries/{query_id}/edit", "Saved query updated."),
            status_code=303,
        )

    @app.post("/queries/automatic")
    async def update_automatic_queries(request: Request) -> Response:
        form = await request.form()
        query_ids = [
            value for raw in form.getlist("query_id")
            if (value := _int_or_none(raw)) is not None
        ]
        paper_ids = [
            value for raw in form.getlist("auto_run_papers")
            if (value := _int_or_none(raw)) is not None
        ]
        blog_ids = [
            value for raw in form.getlist("auto_run_blogs")
            if (value := _int_or_none(raw)) is not None
        ]
        with Database(config.paths.database) as db:
            db.update_automatic_queries(
                query_ids,
                paper_query_ids=paper_ids,
                blog_query_ids=blog_ids,
            )
        if request.headers.get("X-Paperwatch-Async") == "1":
            return JSONResponse(
                {"ok": True, "message": "Automatic query selections updated."}
            )
        return RedirectResponse(
            _notice_url("/queries", "Automatic query selections updated."),
            status_code=303,
        )

    @app.post("/queries/{query_id}/delete")
    def delete_query(query_id: int) -> RedirectResponse:
        with Database(config.paths.database) as db:
            query = db.get_saved_query(query_id)
            if query is None:
                raise HTTPException(404, "Saved query not found")
            if bool(query["is_builtin"]):
                return RedirectResponse(
                    _notice_url("/queries", "Summarize is a built-in prompt and cannot be deleted.", level="warning"),
                    status_code=303,
                )
            db.delete_saved_query(query_id)
        return RedirectResponse(_notice_url("/queries", "Saved query deleted."), status_code=303)

    @app.post("/queries/{query_id}/classifications/delete")
    def delete_query_classifications(query_id: int) -> RedirectResponse:
        with Database(config.paths.database) as db:
            query = db.get_saved_query(query_id)
            if query is None:
                raise HTTPException(404, "Saved query not found")
            removed = db.clear_query_classifications(query_id)
        return RedirectResponse(
            _notice_url(
                f"/queries#query-{query_id}",
                f"Removed {removed} classification result{'s' if removed != 1 else ''}.",
            ),
            status_code=303,
        )

    @app.post("/queries/{query_id}/run")
    async def run_query(request: Request, query_id: int) -> RedirectResponse:
        form = await request.form()
        profiles = tuple(str(value) for value in form.getlist("model"))
        try:
            context = _configured_context_window(
                config, profiles, _int_or_none(form.get("context_window_tokens"))
            )
        except ValueError as exc:
            return RedirectResponse(
                _notice_url("/queries", str(exc), level="error"), status_code=303
            )
        source_scope = str(form.get("source_scope", "abstract"))
        pdf_strategy = str(form.get("pdf_strategy", "auto"))
        try:
            batch_id = collection_controller.start(
                query_id,
                profiles,
                context,
                source_scope=source_scope,
                pdf_strategy=pdf_strategy,
            )
        except ValueError as exc:
            return RedirectResponse(
                _notice_url("/queries", str(exc), level="error"), status_code=303
            )
        if batch_id is None:
            return RedirectResponse(
                _notice_url("/queries", "Another collection query is already running.", level="warning"),
                status_code=303,
            )
        return RedirectResponse(
            _notice_url("/queries", f"Started query with {len(set(profiles))} model(s)."),
            status_code=303,
        )

    @app.post("/queries/run-filtered")
    async def run_filtered_query(request: Request) -> RedirectResponse:
        form = await request.form()
        query_id = _int_or_none(form.get("query_id"))
        if query_id is None:
            return RedirectResponse(
                _notice_url("/", "Select a saved query.", level="error"), status_code=303
            )
        override_defaults = str(form.get("override_query_defaults", "")) in {
            "1", "true", "on"
        }
        run_selected = str(form.get("selection_mode", "")) == "selected"
        filters = _parse_filter_values(form)
        content_filter = _content_kind(str(form.get("content") or "all"), "all")
        blog_compatible_filters = not (
            filters.topic_id
            or filters.topic_ids
            or filters.paper_kind
            or filters.scope
            or filters.has_pdf is True
        )
        paper_filter_args = {
            "query": filters.query,
            "title_query": filters.title_query,
            "author_id": filters.author_id,
            "author_ids": filters.author_ids,
            "author_tags": filters.author_tags,
            "content_tags": filters.content_tags,
            "topic_id": filters.topic_id,
            "topic_ids": filters.topic_ids,
            "paper_kind": filters.paper_kind,
            "scope": filters.scope,
            "has_pdf": filters.has_pdf,
            "analyzed": filters.analyzed,
            "starred": filters.starred,
            "read": filters.read,
            "notes": filters.notes,
            "date_from": filters.date_from,
            "date_to": filters.date_to,
            "sort": filters.sort,
        }
        blog_filter_args = {
            "query": filters.query,
            "title_query": filters.title_query,
            "tags": filters.content_tags,
            "author_id": filters.author_id,
            "author_ids": filters.author_ids,
            "author_tags": filters.author_tags,
            "date_from": filters.date_from,
            "date_to": filters.date_to,
            "starred": filters.starred,
            "read": filters.read,
            "notes": filters.notes,
            "analyzed": filters.analyzed,
            "show_stubs": filters.display_stubs,
        }
        with Database(config.paths.database) as db:
            stored_query = db.get_saved_query(query_id)
            if stored_query is None:
                return RedirectResponse(
                    _notice_url("/", "Saved query not found.", level="error"), status_code=303
                )
            query = _query_view(stored_query, config)
            authors_for_labels = db.list_authors_with_tags(tracked_only=True)
            topics_for_labels = db.list_topics()
            paper_match_count = 0
            blog_match_count = 0
            selected_paper_ids: list[int] = []
            selected_blog_ids: list[int] = []
            if run_selected:
                selected_papers, _ = db.search_papers(
                    selected=True, limit=None, offset=0
                )
                selected_paper_ids = [int(row["id"]) for row in selected_papers]
                selected_blogs = db.list_blog_entries(
                    selected=True, show_stubs=True, limit=None, query_view=True
                )
                selected_blog_ids = [int(row["id"]) for row in selected_blogs]
            else:
                if content_filter != "blogs":
                    _, paper_match_count = db.search_papers(
                        **paper_filter_args, limit=0, offset=0
                    )
                if (
                    content_filter != "papers"
                    and blog_compatible_filters
                    and db.has_visible_blog_entries()
                ):
                    blog_match_count = db.count_blog_entries(**blog_filter_args)

        if override_defaults:
            profiles = tuple(
                str(value) for value in form.getlist("model") if str(value).strip()
            )
            source_scope = str(form.get("source_scope", "abstract"))
            pdf_strategy = str(form.get("pdf_strategy", "auto"))
            requested_context = _int_or_none(form.get("context_window_tokens"))
        else:
            profiles = tuple(query["model_profiles"])
            source_scope = str(query["paper_source_scope"] or "auto")
            pdf_strategy = str(query["pdf_strategy"] or "chunked")
            requested_context = None

        local_regexp = bool(query["local_regexp"])
        if local_regexp:
            profiles = ()
        available_profiles = {item.profile for item in config.available_llms}
        if not local_regexp and (
            not profiles or any(name not in available_profiles for name in profiles)
        ):
            return RedirectResponse(
                _notice_url("/", "Select a configured model.", level="error"), status_code=303
            )
        try:
            context = _configured_context_window(config, profiles, requested_context)
        except ValueError as exc:
            return RedirectResponse(
                _notice_url("/", str(exc), level="error"), status_code=303
            )

        targets: list[tuple[str, dict[str, Any]]] = []
        if run_selected:
            if selected_paper_ids and _query_supports(query, "papers"):
                targets.append(
                    (
                        "papers",
                        {
                            "paper_ids": selected_paper_ids,
                            "labels": [f"Selected papers: {len(selected_paper_ids)}"],
                        },
                    )
                )
            if selected_blog_ids and _query_supports(query, "blogs"):
                targets.append(
                    (
                        "blogs",
                        {
                            "blog_entry_ids": selected_blog_ids,
                            "labels": [
                                f"Selected blog entries: {len(selected_blog_ids)}"
                            ],
                        },
                    )
                )
        else:
            base_selection = _selection_with_labels(
                filters, authors=authors_for_labels, topics=topics_for_labels
            )
            if paper_match_count and _query_supports(query, "papers"):
                targets.append(("papers", dict(base_selection)))
            if blog_match_count and _query_supports(query, "blogs"):
                targets.append(("blogs", dict(base_selection)))
        if not targets:
            unsupported_target = "selected content" if run_selected else "current search results"
            return RedirectResponse(
                _notice_url(
                    "/",
                    f"This saved query does not support the {unsupported_target}.",
                    level="error",
                ),
                status_code=303,
            )

        try:
            if len(targets) == 1:
                content_type, selection = targets[0]
                kwargs: dict[str, Any] = {
                    "selection": selection,
                    "source_scope": "blog" if content_type == "blogs" else source_scope,
                    "pdf_strategy": "auto" if content_type == "blogs" else pdf_strategy,
                }
                if "content_type" in inspect.signature(collection_controller.start).parameters:
                    kwargs["content_type"] = content_type
                batch_result: str | tuple[str, ...] | None = collection_controller.start(
                    query_id, profiles, context, **kwargs
                )
            else:
                batch_result = collection_controller.start_on_results(
                    query_id,
                    profiles,
                    context,
                    targets=tuple(targets),
                    source_scope=source_scope,
                    pdf_strategy=pdf_strategy,
                )
        except ValueError as exc:
            return RedirectResponse(_notice_url("/", str(exc), level="error"), status_code=303)
        if batch_result is None:
            return RedirectResponse(
                _notice_url("/", "Another collection query is already running.", level="warning"),
                status_code=303,
            )
        target = _notice_url(
            f"/queries?focus={query_id}",
            (
                "Started saved query on selected content."
                if run_selected
                else "Started saved query on all matching search results."
            ),
        )
        return RedirectResponse(f"{target}#query-{query_id}", status_code=303)

    @app.post("/queries/{query_id}/run-existing/{content_type}")
    async def run_query_on_existing(
        request: Request, query_id: int, content_type: str
    ) -> RedirectResponse:
        if content_type not in {"papers", "blogs"}:
            raise HTTPException(404, "Unknown content type")
        form = await request.form()
        with Database(config.paths.database) as db:
            stored = db.get_saved_query(query_id)
            if stored is None:
                raise HTTPException(404, "Saved query not found")
            query = _query_view(stored, config)
            selection: dict[str, Any] = {
                "content_type": content_type,
                "labels": [f"All existing {content_type}"],
            }
            if content_type == "papers":
                rows, _ = db.search_papers(limit=None, offset=0)
                eligible_ids = [
                    int(row["id"])
                    for row in rows
                    if str(row["abstract"] or "").strip()
                    or str(row["pdf_url"] or "").strip()
                    or str(row["local_pdf_path"] or "").strip()
                ]
                selection["paper_ids"] = eligible_ids
        requested_profiles = tuple(
            str(value) for value in form.getlist("model") if str(value).strip()
        )
        profiles = requested_profiles or tuple(query["model_profiles"])
        if not profiles and not query["local_regexp"]:
            return RedirectResponse(
                _notice_url(
                    f"/queries/{query_id}/edit",
                    "Select at least one configured model before running this query.",
                    level="error",
                ),
                status_code=303,
            )
        source_scope = (
            "blog" if content_type == "blogs" else str(query["paper_source_scope"])
        )
        pdf_strategy = (
            "auto" if content_type == "blogs" else str(query["pdf_strategy"])
        )
        try:
            batch_id = collection_controller.start(
                query_id,
                profiles,
                _configured_context_window(
                    config,
                    profiles,
                    _int_or_none(form.get("context_window_tokens")),
                ),
                selection=selection,
                source_scope=source_scope,
                pdf_strategy=pdf_strategy,
                content_type=content_type,
            )
        except ValueError as exc:
            return RedirectResponse(
                _notice_url("/queries", str(exc), level="error"), status_code=303
            )
        if batch_id is None:
            return RedirectResponse(
                _notice_url(
                    "/queries",
                    "Another saved prompt is already running.",
                    level="warning",
                ),
                status_code=303,
            )
        return RedirectResponse(
            _notice_url(
                "/queries",
                f"Started {query['name']} on all existing {content_type} with "
                f"{len(profiles)} saved model(s).",
            ),
            status_code=303,
        )

    @app.post("/queries/run-blogs")
    async def run_blog_query(request: Request) -> RedirectResponse:
        form = await request.form()
        query_ids = tuple(
            dict.fromkeys(
                query_id
                for raw in form.getlist("query_id")
                if (query_id := _int_or_none(raw)) is not None
            )
        )
        profile = str(form.get("model", "")).strip()
        if not query_ids:
            return RedirectResponse(
                _notice_url("/blogs", "Select a saved query.", level="error"),
                status_code=303,
            )
        configured_profile = config.llm_profile(profile)
        if configured_profile is None:
            return RedirectResponse(
                _notice_url("/", "Select a configured model.", level="error"),
                status_code=303,
            )
        with Database(config.paths.database) as db:
            selected_ids = [
                int(row["id"])
                for row in db.list_blog_entries(
                    limit=None, selected=True, ids_only=True
                )
            ][:5000]
        if not selected_ids:
            return RedirectResponse(
                _notice_url("/", "Select at least one blog entry.", level="error"),
                status_code=303,
            )
        context = configured_profile.context_window_tokens
        selection = {
            "content_type": "blogs",
            "blog_entry_ids": selected_ids,
            "labels": [f"Selected blog entries: {len(selected_ids)}"],
        }
        try:
            start_many = getattr(collection_controller, "start_many", None)
            if callable(start_many):
                batch_id = start_many(
                    query_ids,
                    profile,
                    context,
                    selection=selection,
                    content_type="blogs",
                )
            else:
                batch_id = collection_controller.start(
                    query_ids[0],
                    (profile,) if profile else (),
                    context,
                    selection=selection,
                    source_scope="blog",
                    pdf_strategy="auto",
                    content_type="blogs",
                )
        except ValueError as exc:
            return RedirectResponse(
                _notice_url("/blogs", str(exc), level="error"), status_code=303
            )
        if batch_id is None:
            return RedirectResponse(
                _notice_url(
                    "/blogs", "Another saved prompt is already running.", level="warning"
                ),
                status_code=303,
            )
        return RedirectResponse(
            _notice_url(
                "/",
                f"Started {len(query_ids)} saved "
                f"quer{'y' if len(query_ids) == 1 else 'ies'} on "
                f"{len(selected_ids)} selected blog "
                f"entr{'y' if len(selected_ids) == 1 else 'ies'}.",
            ),
            status_code=303,
        )

    @app.get("/api/query-status")
    def query_status() -> JSONResponse:
        return JSONResponse(asdict(collection_controller.snapshot()))

    @app.post("/api/query-cancel")
    def cancel_query() -> JSONResponse:
        cancelled = collection_controller.cancel()
        return JSONResponse(
            {"ok": cancelled, "message": "Cancellation requested" if cancelled else "No saved prompt is running"},
            status_code=200 if cancelled else 409,
        )

    @app.get("/api/running-status")
    def running_status() -> JSONResponse:
        return JSONResponse(
            {
                "fetch": asdict(controller.snapshot()),
                "analysis": asdict(model_controller.snapshot()),
                "query": asdict(collection_controller.snapshot()),
            }
        )

    def _data_actions_available() -> bool:
        return not (
            controller.snapshot().running
            or model_controller.snapshot().running
            or collection_controller.snapshot().running
        )

    @app.post("/data/delete/authors")
    def delete_all_authors() -> RedirectResponse:
        if not _data_actions_available():
            return RedirectResponse(
                _notice_url("/runs", "Stop active jobs before deleting data.", level="warning"),
                status_code=303,
            )
        with Database(config.paths.database) as db:
            count = db.delete_all_authors()
        return RedirectResponse(
            _notice_url("/runs", f"Deleted {count} author record(s)."), status_code=303
        )

    @app.post("/data/delete/papers")
    def delete_all_papers() -> RedirectResponse:
        if not _data_actions_available():
            return RedirectResponse(
                _notice_url("/runs", "Stop active jobs before deleting data.", level="warning"),
                status_code=303,
            )
        with Database(config.paths.database) as db:
            count, paths = db.delete_all_papers()
        root = config.paths.pdf_dir.resolve()
        removed = 0
        for value in paths:
            path = Path(value).expanduser().resolve()
            if path.is_file() and (path == root or root in path.parents):
                path.unlink(missing_ok=True)
                removed += 1
        return RedirectResponse(
            _notice_url(
                "/runs", f"Deleted {count} paper record(s) and {removed} stored PDF(s)."
            ),
            status_code=303,
        )

    @app.post("/data/delete/blogs")
    def delete_all_downloaded_blogs() -> RedirectResponse:
        if not _data_actions_available():
            return RedirectResponse(
                _notice_url("/runs", "Stop active jobs before deleting data.", level="warning"),
                status_code=303,
            )
        with Database(config.paths.database) as db:
            count = db.delete_all_blog_entries()
        return RedirectResponse(
            _notice_url(
                "/runs",
                f"Deleted {count} downloaded blog entr{'y' if count == 1 else 'ies'}; source links were kept.",
            ),
            status_code=303,
        )

    @app.post("/data/delete/summaries")
    def delete_all_summaries() -> RedirectResponse:
        if not _data_actions_available():
            return RedirectResponse(
                _notice_url("/runs", "Stop active jobs before deleting data.", level="warning"),
                status_code=303,
            )
        with Database(config.paths.database) as db:
            count = db.delete_all_summaries()
        return RedirectResponse(
            _notice_url(
                "/runs",
                f"Deleted {count} generated summar{'y' if count == 1 else 'ies'} and their task history.",
            ),
            status_code=303,
        )

    @app.post("/data/delete/fetch-history")
    def delete_fetch_history() -> RedirectResponse:
        if not _data_actions_available():
            return RedirectResponse(
                _notice_url("/runs", "Stop active jobs before deleting data.", level="warning"),
                status_code=303,
            )
        with Database(config.paths.database) as db:
            count = db.delete_fetch_history()
        return RedirectResponse(
            _notice_url(
                "/runs",
                f"Deleted {count} fetch-history record{'s' if count != 1 else ''}.",
            ),
            status_code=303,
        )

    @app.post("/data/queries/defaults")
    def install_default_queries() -> RedirectResponse:
        model_profile = config.available_llms[0].profile if config.available_llms else ""
        with Database(config.paths.database) as db:
            count = add_default_queries(db, model_profile)
        message = (
            f"Added {count} default quer{'y' if count == 1 else 'ies'}; automatic runs remain disabled."
            if count
            else "All default queries are already present."
        )
        return RedirectResponse(_notice_url("/queries", message), status_code=303)

    @app.post("/data/delete/queries")
    def delete_all_queries() -> RedirectResponse:
        if not _data_actions_available():
            return RedirectResponse(
                _notice_url("/runs", "Stop active jobs before deleting data.", level="warning"),
                status_code=303,
            )
        with Database(config.paths.database) as db:
            count = db.delete_all_queries()
        return RedirectResponse(
            _notice_url(
                "/runs",
                f"Deleted {count} saved quer{'y' if count == 1 else 'ies'}; the built-in query was restored.",
            ),
            status_code=303,
        )

    @app.post("/data/delete/query-history")
    def delete_query_history() -> RedirectResponse:
        if not _data_actions_available():
            return RedirectResponse(
                _notice_url("/runs", "Stop active jobs before deleting data.", level="warning"),
                status_code=303,
            )
        with Database(config.paths.database) as db:
            count = db.delete_query_history()
        return RedirectResponse(
            _notice_url(
                "/runs",
                f"Deleted {count} saved-query output record{'s' if count != 1 else ''}.",
            ),
            status_code=303,
        )

    @app.get("/data/export/authors-blogs")
    def export_authors_blogs() -> JSONResponse:
        with Database(config.paths.database) as db:
            payload = db.export_author_blog_catalog()
        return JSONResponse(
            payload,
            headers={
                "Content-Disposition": 'attachment; filename="paperwatch-authors-blogs.json"'
            },
        )

    @app.post("/data/import/authors-blogs")
    async def import_authors_blogs(request: Request) -> RedirectResponse:
        if not _data_actions_available():
            return RedirectResponse(
                _notice_url("/runs", "Stop active jobs before importing data.", level="warning"),
                status_code=303,
            )
        form = await request.form()
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            return RedirectResponse(
                _notice_url("/runs", "Choose an authors and blogs JSON file.", level="error"),
                status_code=303,
            )
        try:
            raw = await upload.read(20_000_001)
            if len(raw) > 20_000_000:
                raise ValueError("Authors and blogs import exceeds 20 MB")
            payload = json.loads(raw.decode("utf-8-sig"))
            if not isinstance(payload, dict):
                raise ValueError("Authors and blogs import must be a JSON object")
            with Database(config.paths.database) as db:
                authors_count, links_count = db.import_author_blog_catalog(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            return RedirectResponse(
                _notice_url("/runs", str(exc), level="error"), status_code=303
            )
        finally:
            await upload.close()
        return RedirectResponse(
            _notice_url(
                "/runs",
                f"Imported {authors_count} author(s) and {links_count} author-blog link(s).",
            ),
            status_code=303,
        )

    @app.get("/data/export/all")
    def export_all_data() -> FileResponse:
        handle, archive_name = tempfile.mkstemp(prefix="paperwatch-export-", suffix=".zip")
        os.close(handle)
        archive = Path(archive_name)
        with tempfile.TemporaryDirectory(prefix="paperwatch-export-db-") as temporary:
            database_copy = Path(temporary) / "database.sqlite3"
            with Database(config.paths.database) as db:
                db.backup_to(database_copy)
            pdf_root = config.paths.pdf_dir.resolve()
            pdf_paths: dict[str, str] = {}
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.write(database_copy, "database.sqlite3")
                if pdf_root.is_dir():
                    for file in sorted(pdf_root.rglob("*")):
                        if not file.is_file() or file.is_symlink():
                            continue
                        relative = file.relative_to(pdf_root).as_posix()
                        bundle.write(file, f"pdfs/{relative}")
                        pdf_paths[str(file.resolve())] = relative
                bundle.writestr(
                    "manifest.json",
                    json.dumps(
                        {
                            "format": "paperwatch-full-data",
                            "version": 1,
                            "created_at": _utcnow(),
                            "pdf_paths": pdf_paths,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                )
        return FileResponse(
            archive,
            media_type="application/zip",
            filename="paperwatch-full-data.zip",
            background=BackgroundTask(archive.unlink, missing_ok=True),
        )

    @app.post("/data/import/all")
    async def import_all_data(request: Request) -> RedirectResponse:
        if not _data_actions_available():
            return RedirectResponse(
                _notice_url("/runs", "Stop active jobs before importing data.", level="warning"),
                status_code=303,
            )
        form = await request.form()
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            return RedirectResponse(
                _notice_url("/runs", "Choose a Paperwatch full-data ZIP file.", level="error"),
                status_code=303,
            )
        staging_root = Path(tempfile.mkdtemp(prefix="paperwatch-import-"))
        uploaded = staging_root / "upload.zip"
        try:
            total = 0
            with uploaded.open("wb") as handle:
                while chunk := await upload.read(1024 * 1024):
                    total += len(chunk)
                    if total > 5_000_000_000:
                        raise ValueError("Full-data import exceeds 5 GB")
                    handle.write(chunk)
            if not zipfile.is_zipfile(uploaded):
                raise ValueError("Full-data import is not a ZIP archive")
            database_copy = staging_root / "database.sqlite3"
            pdf_stage = staging_root / "pdfs"
            pdf_stage.mkdir()
            with zipfile.ZipFile(uploaded) as bundle:
                names = set(bundle.namelist())
                if "database.sqlite3" not in names or "manifest.json" not in names:
                    raise ValueError("Full-data archive is missing its database or manifest")
                manifest = json.loads(bundle.read("manifest.json").decode("utf-8"))
                if not isinstance(manifest, dict) or manifest.get("format") != "paperwatch-full-data":
                    raise ValueError("Unsupported full-data archive")
                database_copy.write_bytes(bundle.read("database.sqlite3"))
                for name in names:
                    if not name.startswith("pdfs/") or name.endswith("/"):
                        continue
                    relative = Path(name.removeprefix("pdfs/"))
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("Full-data archive contains an unsafe path")
                    destination = pdf_stage / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(name) as source, destination.open("wb") as target:
                        shutil.copyfileobj(source, target)
            check = sqlite3.connect(str(database_copy))
            try:
                result = check.execute("PRAGMA quick_check").fetchone()
                tables = {
                    str(row[0])
                    for row in check.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
            finally:
                check.close()
            if not result or result[0] != "ok" or not {"authors", "papers", "blogs", "blog_entries"} <= tables:
                raise ValueError("Full-data database is invalid")
            with Database(database_copy):
                pass
            pdf_mapping = manifest.get("pdf_paths") if isinstance(manifest.get("pdf_paths"), dict) else {}
            if pdf_mapping:
                conn = sqlite3.connect(str(database_copy))
                try:
                    for old_path, relative in pdf_mapping.items():
                        relative_path = Path(str(relative))
                        if relative_path.is_absolute() or ".." in relative_path.parts:
                            continue
                        destination = config.paths.pdf_dir.resolve() / relative_path
                        conn.execute(
                            "UPDATE papers SET local_pdf_path=? WHERE local_pdf_path=?",
                            (str(destination), str(old_path)),
                        )
                    conn.commit()
                finally:
                    conn.close()
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            database_target = config.paths.database.resolve()
            database_target.parent.mkdir(parents=True, exist_ok=True)
            if database_target.exists():
                backup = database_target.with_name(f"{database_target.name}.before-import-{stamp}")
                shutil.copy2(database_target, backup)
            for suffix in ("-wal", "-shm"):
                Path(str(database_target) + suffix).unlink(missing_ok=True)
            os.replace(database_copy, database_target)
            pdf_target = config.paths.pdf_dir.resolve()
            if pdf_target.exists():
                pdf_backup = pdf_target.with_name(f"{pdf_target.name}.before-import-{stamp}")
                if pdf_backup.exists():
                    shutil.rmtree(pdf_backup)
                os.replace(pdf_target, pdf_backup)
            os.replace(pdf_stage, pdf_target)
        except (OSError, sqlite3.Error, zipfile.BadZipFile, json.JSONDecodeError, ValueError) as exc:
            return RedirectResponse(
                _notice_url("/runs", f"Import failed: {exc}", level="error"), status_code=303
            )
        finally:
            await upload.close()
            shutil.rmtree(staging_root, ignore_errors=True)
        return RedirectResponse(
            _notice_url("/runs", "Imported the full Paperwatch database and PDF store."),
            status_code=303,
        )

    @app.get("/runs", response_class=HTMLResponse)
    def runs(request: Request) -> HTMLResponse:
        active_jobs: list[dict[str, Any]] = []
        fetch_snapshot = controller.snapshot()
        analysis_snapshot = model_controller.snapshot()
        query_snapshot = collection_controller.snapshot()
        if fetch_snapshot.running:
            active_jobs.append(
                {
                    "kind": "Fetch",
                    "live_kind": "fetch",
                    "status": "running",
                    "name": fetch_snapshot.stage or "Fetch now",
                    "model": "—",
                    "papers": [],
                    "blogs": [],
                    "started_at": fetch_snapshot.started_at,
                    "progress_current": fetch_snapshot.progress_current,
                    "progress_total": fetch_snapshot.progress_total,
                    "message": fetch_snapshot.message,
                }
            )
        with Database(config.paths.database) as db:
            rows = db.list_runs(100)
            for task in db.active_analysis_tasks():
                live = (
                    analysis_snapshot.running
                    and str(task["batch_id"]) == str(analysis_snapshot.batch_id)
                )
                active_jobs.append(
                    {
                        "kind": "Summary",
                        "live_kind": "analysis" if live else "",
                        "status": task["status"],
                        "name": f"{task['scope']} analysis",
                        "model": task["model"],
                        "paper_id": int(task["paper_id"]),
                        "paper_title": task["paper_title"],
                        "papers": [{"id": int(task["paper_id"]), "title": task["paper_title"]}],
                        "started_at": task["started_at"] or task["created_at"],
                        "progress_current": analysis_snapshot.progress_current if live else 0,
                        "progress_total": analysis_snapshot.progress_total if live else 0,
                        "message": analysis_snapshot.message if live else "",
                    }
                )
            for query_run in db.active_query_runs():
                live = (
                    query_snapshot.running
                    and str(query_run["batch_id"]) == str(query_snapshot.batch_id)
                )
                paper_id: int | None = None
                paper_title: str | None = None
                papers_for_job: list[dict[str, Any]] = []
                blogs_for_job: list[dict[str, Any]] = []
                selection = _json_value(query_run["selection_json"], {})
                values = selection.get("paper_ids") if isinstance(selection, dict) else None
                if isinstance(values, list):
                    for value in values[:20]:
                        selected_id = _int_or_none(value)
                        paper = db.get_paper(selected_id) if selected_id is not None else None
                        if paper is not None:
                            papers_for_job.append({"id": selected_id, "title": str(paper["title"])})
                    if len(papers_for_job) == 1:
                        paper_id = int(papers_for_job[0]["id"])
                        paper_title = str(papers_for_job[0]["title"])
                blog_values = (
                    selection.get("blog_entry_ids")
                    if isinstance(selection, dict)
                    else None
                )
                if isinstance(blog_values, list):
                    selected_blog_ids = [
                        value
                        for value in (_int_or_none(item) for item in blog_values[:20])
                        if value is not None
                    ]
                    for entry in db.blog_entries_by_ids(selected_blog_ids):
                        blogs_for_job.append(
                            {"id": int(entry["id"]), "title": str(entry["title"])}
                        )
                active_jobs.append(
                    {
                        "kind": "Saved prompt",
                        "live_kind": "query" if live else "",
                        "status": query_run["status"],
                        "name": query_run["query_name"],
                        "model": query_run["model"],
                        "paper_id": paper_id,
                        "paper_title": paper_title,
                        "papers": papers_for_job,
                        "blogs": blogs_for_job,
                        "started_at": query_run["started_at"] or query_run["created_at"],
                        "progress_current": query_snapshot.progress_current if live else 0,
                        "progress_total": query_snapshot.progress_total if live else 0,
                        "message": query_snapshot.message if live else "",
                    }
                )
        return render(request, "runs.html", runs=rows, active_jobs=active_jobs)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_detail(request: Request, run_id: int) -> HTMLResponse:
        with Database(config.paths.database) as db:
            run = db.get_run(run_id)
            if run is None:
                raise HTTPException(404, "Run not found")
            errors = db.get_run_errors(run_id)
        return render(request, "run_detail.html", run=run, errors=errors)

    return app
