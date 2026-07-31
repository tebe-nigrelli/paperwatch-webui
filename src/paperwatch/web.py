from __future__ import annotations

import csv
import inspect
import io
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

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.datastructures import UploadFile

from .analysis_jobs import AnalysisController
from .config import AppConfig, load_config
from .db import Database
from .locking import AnalysisAlreadyInProgress, analysis_lock
from .models import (
    AuthorCandidate,
    PaperAuthor,
    PaperRecord,
    clean_text,
    extract_arxiv_id,
    short_openalex_id,
    strip_doi,
)
from .pdfs import PDFError, safe_pdf_name, validate_pdf_file
from .query_jobs import QueryController
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
DEFAULT_PAGE_SIZE = 100
DEFAULT_CONTEXT_WINDOW_TOKENS = 16384


@dataclass(frozen=True, slots=True)
class PaperFilters:
    query: str = ""
    author_id: str = ""
    topic_id: str = ""
    paper_kind: str = ""
    scope: str = ""
    has_pdf: bool | None = None
    analyzed: bool | None = None
    date_from: str = ""
    date_to: str = ""
    sort: str = "newest"
    page: int = 1
    page_size: int = DEFAULT_PAGE_SIZE

    def selection(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "author_id": self.author_id,
            "topic_id": self.topic_id,
            "paper_kind": self.paper_kind,
            "scope": self.scope,
            "has_pdf": self.has_pdf,
            "analyzed": self.analyzed,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "sort": self.sort,
        }

    def selection_items(self) -> list[tuple[str, str]]:
        values = {
            "q": self.query,
            "author": self.author_id,
            "topic": self.topic_id,
            "kind": self.paper_kind,
            "scope": self.scope,
            "has_pdf": _bool_query(self.has_pdf),
            "analyzed": _bool_query(self.analyzed),
            "date_from": self.date_from,
            "date_to": self.date_to,
            "sort": self.sort if self.sort != "newest" else "",
        }
        return [(key, value) for key, value in values.items() if value]

    def query_items(self, *, page: int | None = None) -> list[tuple[str, str]]:
        values = {
            "q": self.query,
            "author": self.author_id,
            "topic": self.topic_id,
            "kind": self.paper_kind,
            "scope": self.scope,
            "has_pdf": _bool_query(self.has_pdf),
            "analyzed": _bool_query(self.analyzed),
            "date_from": self.date_from,
            "date_to": self.date_to,
            "sort": self.sort if self.sort != "newest" else "",
            "page_size": str(self.page_size) if self.page_size != DEFAULT_PAGE_SIZE else "",
            "page": str(page if page is not None else self.page),
        }
        return [(key, value) for key, value in values.items() if value and value != "1"]

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

    def start(self, mode: str = "run", query: str | None = None) -> bool:
        if mode not in {"run", "backfill", "search", "download"}:
            raise ValueError("Unknown run mode")
        normalized_query = query.strip() if query else None
        if mode in {"search", "download"} and not normalized_query:
            raise ValueError("Paper identifier cannot be empty")
        with self._lock:
            if self._snapshot.running:
                return False
            self._snapshot = RunSnapshot(
                running=True,
                mode=mode,
                query=normalized_query,
                started_at=_utcnow(),
                status="running",
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
            if parameter_count >= 3:
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
        except Exception as exc:
            with self._lock:
                self._snapshot.running = False
                self._snapshot.finished_at = _utcnow()
                self._snapshot.status = "failed"
                self._snapshot.error = str(exc)

    def snapshot(self) -> RunSnapshot:
        with self._lock:
            return RunSnapshot(**asdict(self._snapshot))

    def wait(self, timeout: float = 10) -> None:
        thread = self._thread
        if thread:
            thread.join(timeout)


def _run_pipeline(
    config_path: Path, mode: str = "run", query: str | None = None
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
    topic_id = short_openalex_id(params.get("topic", "").strip()[:64]) or ""
    if topic_id and not topic_id.startswith("T"):
        topic_id = ""
    return PaperFilters(
        query=params.get("q", "").strip()[:200],
        author_id=params.get("author", "").strip()[:64],
        topic_id=topic_id,
        paper_kind=paper_kind,
        scope=scope,
        has_pdf=_parse_tristate(params.get("has_pdf")),
        analyzed=_parse_tristate(params.get("analyzed")),
        date_from=_valid_date(params.get("date_from")),
        date_to=_valid_date(params.get("date_to")),
        sort=sort,
        page=page,
        page_size=page_size,
    )


def _parse_filters(request: Request) -> PaperFilters:
    return _parse_filter_values(request.query_params)


def _json_value(value: Any, fallback: Any) -> Any:
    if value in (None, ""):
        return fallback
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def _row_dict(row: Any) -> dict[str, Any]:
    return dict(row) if row is not None else {}


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


def _all_filtered(db: Database, filters: PaperFilters) -> list[Any]:
    rows, _ = db.search_papers(
        query=filters.query,
        author_id=filters.author_id,
        topic_id=filters.topic_id,
        paper_kind=filters.paper_kind,
        scope=filters.scope,
        has_pdf=filters.has_pdf,
        analyzed=filters.analyzed,
        date_from=filters.date_from,
        date_to=filters.date_to,
        sort=filters.sort,
        limit=None,
        offset=0,
    )
    return rows


def _notice_url(path: str, message: str, *, level: str = "success") -> str:
    separator = "&" if "?" in path else "?"
    return f"{path}{separator}{urlencode({'notice': message, 'level': level})}"


def _author_client(config: AppConfig) -> OpenAlexClient:
    return OpenAlexClient(
        api_key=config.openalex.api_key,
        base_url=config.openalex.base_url,
        timeout_seconds=config.openalex.timeout_seconds,
    )


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


def _paper_table_view(db: Database, row: Any) -> dict[str, Any]:
    data = _paper_view(row)
    paper_authors = [_author_view(item) for item in db.paper_authors(int(data["id"]))]
    tracked_authors = [item for item in paper_authors if bool(item.get("is_tracked"))]
    data["tracked_authors"] = tracked_authors
    data["other_author_count"] = max(0, len(paper_authors) - len(tracked_authors))
    data["all_authors"] = paper_authors
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
    source_scope = str(data["selection"].get("source_scope") or "abstract")
    pdf_strategy = str(data["selection"].get("pdf_strategy") or "auto")
    selection_items.append(f"Source: {source_scope}")
    if source_scope == "fulltext":
        selection_items.append(f"PDF strategy: {pdf_strategy}")
    data["selection_label"] = " · ".join(selection_items) or "All matching records"
    selection_query = urlencode([
        (key, value)
        for key, value in {
            "q": data["selection"].get("query"),
            "author": data["selection"].get("author_id"),
            "topic": data["selection"].get("topic_id"),
            "kind": data["selection"].get("paper_kind"),
            "scope": data["selection"].get("scope"),
            "has_pdf": _bool_query(data["selection"].get("has_pdf")),
            "analyzed": _bool_query(data["selection"].get("analyzed")),
            "date_from": data["selection"].get("date_from"),
            "date_to": data["selection"].get("date_to"),
            "sort": data["selection"].get("sort")
            if data["selection"].get("sort") != "newest"
            else "",
        }.items()
        if value
    ])
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
) -> FastAPI:
    config = load_config(config_path)
    controller = run_controller or RunController(config.config_path)
    model_controller = analysis_controller or AnalysisController(config.config_path)
    collection_controller = query_controller or QueryController(config.config_path)
    client_factory = author_client_factory or _author_client
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

    app = FastAPI(title="paperwatch", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.config = config
    app.state.run_controller = controller
    app.state.analysis_controller = model_controller
    app.state.query_controller = collection_controller
    app.state.author_client_factory = client_factory
    app.mount("/static", StaticFiles(directory=str(package_dir / "web_static")), name="static")

    def render(request: Request, template: str, **context: Any) -> HTMLResponse:
        base = {
            "request": request,
            "config": config,
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
            "default-src 'self'; style-src 'self'; img-src 'self' data:; "
            "script-src 'self'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
        )
        return response

    @app.get("/", response_class=HTMLResponse)
    def recent_papers(request: Request) -> HTMLResponse:
        filters = _parse_filters(request)
        offset = (filters.page - 1) * filters.page_size
        with Database(config.paths.database) as db:
            rows, total = db.search_papers(
                query=filters.query,
                author_id=filters.author_id,
                topic_id=filters.topic_id,
                paper_kind=filters.paper_kind,
                scope=filters.scope,
                has_pdf=filters.has_pdf,
                analyzed=filters.analyzed,
                date_from=filters.date_from,
                date_to=filters.date_to,
                sort=filters.sort,
                limit=filters.page_size,
                offset=offset,
            )
            author_rows = db.list_authors_with_tags(tracked_only=True)
            topic_rows = db.list_topics()
            saved_queries = [dict(row) for row in db.list_saved_queries()]
            recent_errors = db.recent_errors(3)
            paper_views = [_paper_table_view(db, row) for row in rows]
        pages = max(1, (total + filters.page_size - 1) // filters.page_size)
        return render(
            request,
            "papers.html",
            papers=paper_views,
            filters=filters,
            authors=author_rows,
            topics=topic_rows,
            saved_queries=saved_queries,
            model_profiles=config.available_llms,
            total=total,
            pages=pages,
            is_home=True,
            recent_errors=recent_errors,
        )

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard(request: Request) -> HTMLResponse:
        with Database(config.paths.database) as db:
            stats = db.dashboard_stats()
            papers, _ = db.search_papers(limit=8, offset=0)
            runs = db.list_runs(8)
            errors = db.recent_errors(6)
        return render(
            request,
            "dashboard.html",
            stats=stats,
            papers=papers,
            runs=runs,
            errors=errors,
        )

    @app.get("/authors", response_class=HTMLResponse)
    def authors(request: Request) -> HTMLResponse:
        tag = request.query_params.get("tag", "").strip()[:40]
        name = request.query_params.get("name", "").strip()[:200]
        query = request.query_params.get("q", "").strip()[:200]
        with Database(config.paths.database) as db:
            rows = db.list_authors_with_tags(
                tracked_only=True, tag=tag, query=query
            )
            tags = db.list_tags()
        return render(
            request,
            "authors.html",
            authors=rows,
            tags=tags,
            active_tag=tag,
            author_query=query,
        )

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

    @app.post("/authors/{openalex_id}/enable")
    def enable_author(openalex_id: str) -> RedirectResponse:
        with Database(config.paths.database) as db:
            if not db.set_author_enabled(openalex_id, True):
                raise HTTPException(404, "Author not found")
        return RedirectResponse(_notice_url("/authors", "Author enabled."), status_code=303)

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
    async def add_author_tag(request: Request, openalex_id: str) -> RedirectResponse:
        form = await request.form()
        tag = str(form.get("tag", "")).strip()
        try:
            with Database(config.paths.database) as db:
                if not db.add_author_tag(openalex_id, tag):
                    raise HTTPException(404, "Author not found")
        except ValueError as exc:
            return RedirectResponse(
                _notice_url("/authors", str(exc), level="error"), status_code=303
            )
        return RedirectResponse(_notice_url("/authors", f"Added tag {tag}."), status_code=303)

    @app.post("/authors/{openalex_id}/tags/remove")
    async def remove_author_tag(request: Request, openalex_id: str) -> RedirectResponse:
        form = await request.form()
        tag = str(form.get("tag", "")).strip()
        with Database(config.paths.database) as db:
            db.remove_author_tag(openalex_id, tag)
        return RedirectResponse(_notice_url("/authors", f"Removed tag {tag}."), status_code=303)

    @app.get("/papers/import", response_class=HTMLResponse)
    def import_paper_form(request: Request) -> HTMLResponse:
        return render(
            request,
            "import_paper.html",
            import_values={},
            import_error="",
        )

    @app.post("/papers/import", response_class=HTMLResponse)
    async def import_paper(request: Request) -> Response:
        values: dict[str, str] = {}
        staged: Path | None = None
        backup: Path | None = None
        destination: Path | None = None
        upload: UploadFile | None = None
        try:
            form = await request.form(
                max_files=1,
                max_fields=20,
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
                )
            }
            form_upload = form.get("pdf")
            if not isinstance(form_upload, UploadFile) or not form_upload.filename:
                raise ValueError("PDF file is required")
            upload = form_upload
            paper = _manual_paper_record(values, filename=upload.filename)
            staged = await _stage_uploaded_pdf(
                upload,
                directory=config.paths.pdf_dir,
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
                db.set_local_pdf(paper_id, str(destination))
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
                import_values=values,
                import_error=str(exc),
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
        if requested_mode not in {"paper", "author"}:
            requested_mode = "author" if legacy_author_query and not legacy_paper_query else "paper"
        query_limit = 500 if requested_mode == "paper" else 200
        query = request.query_params.get("q", "").strip()[:query_limit]
        if not query:
            query = legacy_paper_query if requested_mode == "paper" else legacy_author_query
        paper_query = query if requested_mode == "paper" else ""
        author_query = query if requested_mode == "author" else ""
        limit = _int_or_none(request.query_params.get("limit")) or 100
        limit = min(max(limit, 1), 100)
        paper_results: list[PaperRecord] = []
        author_results: list[AuthorCandidate] = []
        paper_error = ""
        author_error = ""
        if paper_query or author_query:
            client = client_factory(config)
            try:
                if paper_query:
                    try:
                        paper_results = client.search_works(paper_query, limit=limit)
                    except Exception as exc:
                        paper_error = str(exc)
                if author_query:
                    try:
                        author_results = client.search_authors(author_query, limit=limit)
                    except Exception as exc:
                        author_error = str(exc)
            finally:
                client.close()
        return render(
            request,
            "discover.html",
            discover_mode=requested_mode,
            discover_query=query,
            paper_query=paper_query,
            author_query=author_query,
            search_limit=limit,
            paper_results=paper_results,
            author_results=author_results,
            paper_error=paper_error,
            author_error=author_error,
        )

    @app.get("/search")
    def search_papers(request: Request) -> RedirectResponse:
        query = request.query_params.get("q", "").strip()[:500]
        target = f"/discover?mode=paper&q={quote_plus(query)}" if query else "/discover?mode=paper"
        return RedirectResponse(target, status_code=303)

    @app.post("/actions/run")
    def start_run() -> RedirectResponse:
        started = controller.start("run")
        if started:
            message = "Fetch started. This page updates while it runs."
            level = "success"
        else:
            message = "A fetch is already running."
            level = "warning"
        return RedirectResponse(_notice_url("/", message, level=level), status_code=303)

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
            message = f"Downloading {title or work_id}."
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
        filters = _parse_filters(request)
        offset = (filters.page - 1) * filters.page_size
        with Database(config.paths.database) as db:
            rows, total = db.search_papers(
                query=filters.query,
                author_id=filters.author_id,
                topic_id=filters.topic_id,
                paper_kind=filters.paper_kind,
                scope=filters.scope,
                has_pdf=filters.has_pdf,
                analyzed=filters.analyzed,
                date_from=filters.date_from,
                date_to=filters.date_to,
                sort=filters.sort,
                limit=filters.page_size,
                offset=offset,
            )
            author_rows = db.list_authors(tracked_only=True)
            topic_rows = db.list_topics()
            saved_queries = [dict(row) for row in db.list_saved_queries()]
            paper_views = [_paper_table_view(db, row) for row in rows]
        pages = max(1, (total + filters.page_size - 1) // filters.page_size)
        if filters.page > pages and total:
            target = urlencode(filters.query_items(page=pages))
            return RedirectResponse(f"/papers?{target}", status_code=303)
        return render(
            request,
            "papers.html",
            papers=paper_views,
            filters=filters,
            authors=author_rows,
            topics=topic_rows,
            saved_queries=saved_queries,
            model_profiles=config.available_llms,
            total=total,
            pages=pages,
        )

    @app.get("/papers/{paper_id}", response_class=HTMLResponse)
    def paper_detail(request: Request, paper_id: int) -> HTMLResponse:
        with Database(config.paths.database) as db:
            paper_row = db.get_paper(paper_id)
            if paper_row is None:
                raise HTTPException(404, "Paper not found")
            paper = _paper_view(paper_row)
            author_rows = [_author_view(row) for row in db.paper_authors(paper_id)]
            default_author_index = next(
                (index for index, author in enumerate(author_rows) if bool(author.get("is_tracked"))),
                0,
            )
            topic_rows = [dict(row) for row in db.paper_topics(paper_id)]
            analysis_tasks = [
                dict(row) for row in db.visible_analysis_tasks_for_paper(paper_id)
            ]
            saved_queries = [dict(row) for row in db.list_saved_queries()]
            prompt_runs = [
                _query_output(row) for row in db.query_runs_for_paper(paper_id)
            ]
        return render(
            request,
            "paper_detail.html",
            paper=paper,
            authors=author_rows,
            default_author_index=default_author_index,
            topics=topic_rows,
            analysis_tasks=analysis_tasks,
            saved_queries=saved_queries,
            prompt_runs=prompt_runs,
            model_profiles=config.available_llms,
            metadata=_json_value(paper["metadata_json"], {}),
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

    @app.post("/papers/{paper_id}/analyze")
    async def analyze_paper(request: Request, paper_id: int) -> Response:
        async_request = request.headers.get("x-paperwatch-async") == "1"
        form = await request.form()
        scope = str(form.get("scope", "fulltext"))
        if scope not in SCOPES:
            raise HTTPException(400, "Invalid analysis scope")
        profile_names = tuple(str(value) for value in form.getlist("model"))
        strategy = str(form.get("strategy", "auto"))
        context_tokens = _int_or_none(form.get("context_window_tokens"))
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
        profiles = tuple(str(value) for value in form.getlist("model") if str(value))
        context = _int_or_none(form.get("context_window_tokens")) or DEFAULT_CONTEXT_WINDOW_TOKENS
        source_scope = str(form.get("source_scope", "abstract"))
        pdf_strategy = str(form.get("pdf_strategy", "auto"))
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
        if source_scope == "abstract" and not paper["abstract"]:
            return failure("This paper has no abstract to analyze.")
        if source_scope == "fulltext" and not (paper["pdf_url"] or paper["local_pdf_path"]):
            return failure("This paper has no PDF available for full-text analysis.")
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
        message = f"Started {query['name']} with {len(set(profiles))} model(s)."
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
            paper = db.get_paper(paper_id)
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
            records = [_paper_export(db, row) for row in _all_filtered(db, filters)]
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
            saved = [dict(row) for row in db.list_saved_queries()]
            runs = [_query_output(row) for row in db.list_query_runs(limit=100)]
        runs_by_query: dict[int, list[dict[str, Any]]] = {}
        for row in runs:
            runs_by_query.setdefault(int(row["query_id"]), []).append(row)
        return render(
            request,
            "queries.html",
            queries=saved,
            runs_by_query=runs_by_query,
            model_profiles=config.available_llms,
        )

    @app.post("/queries")
    async def create_query(request: Request) -> RedirectResponse:
        form = await request.form()
        try:
            with Database(config.paths.database) as db:
                db.create_saved_query(
                    name=str(form.get("name", "")),
                    prompt=str(form.get("prompt", "")),
                )
        except ValueError as exc:
            return RedirectResponse(
                _notice_url("/queries", str(exc), level="error"), status_code=303
            )
        return RedirectResponse(_notice_url("/queries", "Saved query created."), status_code=303)

    @app.post("/queries/{query_id}/edit")
    async def edit_query(request: Request, query_id: int) -> RedirectResponse:
        form = await request.form()
        try:
            with Database(config.paths.database) as db:
                changed = db.update_saved_query(
                    query_id,
                    name=str(form.get("name", "")),
                    prompt=str(form.get("prompt", "")),
                )
            if not changed:
                raise HTTPException(404, "Saved query not found")
        except ValueError as exc:
            return RedirectResponse(
                _notice_url("/queries", str(exc), level="error"), status_code=303
            )
        return RedirectResponse(_notice_url("/queries", "Saved query updated."), status_code=303)

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

    @app.post("/queries/{query_id}/run")
    async def run_query(request: Request, query_id: int) -> RedirectResponse:
        form = await request.form()
        profiles = tuple(str(value) for value in form.getlist("model"))
        context = _int_or_none(form.get("context_window_tokens")) or DEFAULT_CONTEXT_WINDOW_TOKENS
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
        profiles = tuple(str(value) for value in form.getlist("model") if str(value))
        context = _int_or_none(form.get("context_window_tokens")) or DEFAULT_CONTEXT_WINDOW_TOKENS
        source_scope = str(form.get("source_scope", "abstract"))
        pdf_strategy = str(form.get("pdf_strategy", "auto"))
        if query_id is None:
            return RedirectResponse(
                _notice_url("/", "Select a saved query.", level="error"), status_code=303
            )
        available_profiles = {item.profile for item in config.available_llms}
        if not profiles or any(name not in available_profiles for name in profiles):
            return RedirectResponse(
                _notice_url("/", "Select a configured model.", level="error"), status_code=303
            )
        filters = _parse_filter_values(form)
        with Database(config.paths.database) as db:
            authors_for_labels = db.list_authors_with_tags(tracked_only=True)
            topics_for_labels = db.list_topics()
        selection = _selection_with_labels(
            filters, authors=authors_for_labels, topics=topics_for_labels
        )
        selected_ids: list[int] = []
        raw_ids = str(form.get("paper_ids", ""))
        for value in raw_ids.split(","):
            paper_id = _int_or_none(value.strip())
            if paper_id is not None and paper_id > 0:
                selected_ids.append(paper_id)
        if selected_ids:
            selected_ids = list(dict.fromkeys(selected_ids))[:5000]
            selection["paper_ids"] = selected_ids
            selection["labels"] = [f"Selected papers: {len(selected_ids)}"]
        try:
            batch_id = collection_controller.start(
                query_id,
                profiles,
                context,
                selection=selection,
                source_scope=source_scope,
                pdf_strategy=pdf_strategy,
            )
        except ValueError as exc:
            return RedirectResponse(_notice_url("/", str(exc), level="error"), status_code=303)
        if batch_id is None:
            return RedirectResponse(
                _notice_url("/", "Another collection query is already running.", level="warning"),
                status_code=303,
            )
        target = _notice_url(
            f"/queries?focus={query_id}",
            f"Started saved query with {len(set(profiles))} model(s).",
        )
        return RedirectResponse(f"{target}#query-{query_id}", status_code=303)

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

    @app.get("/runs", response_class=HTMLResponse)
    def runs(request: Request) -> HTMLResponse:
        active_jobs: list[dict[str, Any]] = []
        with Database(config.paths.database) as db:
            rows = db.list_runs(100)
            for task in db.active_analysis_tasks():
                active_jobs.append(
                    {
                        "kind": "Summary",
                        "status": task["status"],
                        "name": f"{task['scope']} analysis",
                        "model": task["model"],
                        "paper_id": int(task["paper_id"]),
                        "paper_title": task["paper_title"],
                        "papers": [{"id": int(task["paper_id"]), "title": task["paper_title"]}],
                        "started_at": task["started_at"] or task["created_at"],
                    }
                )
            for query_run in db.active_query_runs():
                paper_id: int | None = None
                paper_title: str | None = None
                papers_for_job: list[dict[str, Any]] = []
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
                active_jobs.append(
                    {
                        "kind": "Saved prompt",
                        "status": query_run["status"],
                        "name": query_run["query_name"],
                        "model": query_run["model"],
                        "paper_id": paper_id,
                        "paper_title": paper_title,
                        "papers": papers_for_job,
                        "started_at": query_run["started_at"] or query_run["created_at"],
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
