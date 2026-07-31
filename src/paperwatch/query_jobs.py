from __future__ import annotations

import inspect
import json

import httpx
import secrets
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from . import USER_AGENT
from .analysis_jobs import _paper_text
from .chunking import context_char_budget
from .config import load_config
from .db import Database
from .job_runtime import JobCancelled, ProgressCallback, check_cancelled, publish
from .llm import (
    LLMClient, LLMError, configure_context_window, configure_job_runtime, create_llm
)
from .locking import analysis_lock
from .models import BATCH_QUERY_JSON_SCHEMA, BatchQueryResult


@dataclass(slots=True)
class QueryBatchSnapshot:
    running: bool = False
    batch_id: str | None = None
    query_id: int | None = None
    paper_id: int | None = None
    profiles: tuple[str, ...] = ()
    completed: int = 0
    failed: int = 0
    cancelled: bool = False
    cancel_requested: bool = False
    stage: str = ""
    message: str = ""
    live_output: str = ""
    error: str | None = None


def _paper_document(db: Database, row: Any) -> dict[str, Any]:
    return {
        "paper_id": int(row["id"]),
        "title": row["title"],
        "publication_date": row["publication_date"],
        "authors": [item["name"] for item in db.paper_authors(int(row["id"]))],
        "venue": row["venue"],
        "abstract": row["abstract"],
        "doi": row["doi"],
        "arxiv_id": row["arxiv_id"],
        "openalex_id": row["openalex_id"],
    }


def _selection_search_args(selection: dict[str, Any] | None) -> dict[str, Any]:
    selection = selection or {}
    return {
        "query": str(selection.get("query") or ""),
        "author_id": str(selection.get("author_id") or ""),
        "topic_id": str(selection.get("topic_id") or ""),
        "paper_kind": str(selection.get("paper_kind") or ""),
        "scope": str(selection.get("scope") or ""),
        "has_pdf": selection.get("has_pdf") if isinstance(selection.get("has_pdf"), bool) else None,
        "analyzed": selection.get("analyzed") if isinstance(selection.get("analyzed"), bool) else None,
        "date_from": str(selection.get("date_from") or ""),
        "date_to": str(selection.get("date_to") or ""),
        "sort": str(selection.get("sort") or "newest"),
    }


def _selected_papers(db: Database, selection: dict[str, Any] | None) -> list[Any]:
    selected_ids: list[int] = []
    if selection and isinstance(selection.get("paper_ids"), list):
        for value in selection["paper_ids"]:
            try:
                selected_ids.append(int(value))
            except (TypeError, ValueError):
                continue
    if selected_ids:
        return [row for paper_id in dict.fromkeys(selected_ids) if (row := db.get_paper(paper_id)) is not None]
    papers, _ = db.search_papers(**_selection_search_args(selection), limit=None, offset=0)
    return papers


def validate_query_sources(
    db: Database,
    *,
    selection: dict[str, Any] | None,
    source_scope: str,
    pdf_strategy: str,
) -> int:
    """Reject a saved-prompt run when any selected paper lacks its source."""
    papers = _selected_papers(db, selection)
    if not papers:
        raise ValueError("No papers match the current selection")
    if source_scope == "abstract":
        if pdf_strategy != "auto":
            raise ValueError("PDF strategy only applies when the source is Full paper PDF")
        missing = [row for row in papers if not str(row["abstract"] or "").strip()]
        source_label = "an abstract"
    else:
        missing = []
        for row in papers:
            pdf_url = str(row["pdf_url"] or "").strip()
            local_value = str(row["local_pdf_path"] or "").strip()
            local_pdf = Path(local_value).expanduser() if local_value else None
            if not pdf_url and not (local_pdf and local_pdf.is_file()):
                missing.append(row)
        source_label = "a PDF"
    if missing:
        examples = ", ".join(str(row["title"]) for row in missing[:3])
        remainder = len(missing) - min(3, len(missing))
        suffix = f" and {remainder} more" if remainder else ""
        noun = "paper" if len(missing) == 1 else "papers"
        raise ValueError(
            f"Cannot run this saved prompt: {len(missing)} selected {noun} "
            f"do not have {source_label}: {examples}{suffix}"
        )
    return len(papers)


def collect_query_documents(
    db: Database,
    source_scope: str = "abstract",
    search_query: str = "",
    *,
    selection: dict[str, Any] | None = None,
    config: Any | None = None,
    pdf_client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    """Collect paper records using run-time source choices.

    ``papers`` remains an accepted alias for older callers and means abstracts/metadata.
    The legacy ``search_query`` argument is retained only for API compatibility.
    """
    if source_scope == "papers":
        source_scope = "abstract"
    if source_scope not in {"abstract", "fulltext"}:
        raise ValueError("Query source must be abstract or fulltext")
    effective_selection = dict(selection or {})
    if search_query and not selection:
        effective_selection["query"] = search_query
    papers = _selected_papers(db, effective_selection)
    documents: list[dict[str, Any]] = []
    owns_client = False
    if source_scope == "fulltext" and pdf_client is None:
        if config is None:
            raise ValueError("Full-text saved prompts require application configuration")
        pdf_client = httpx.Client(
            timeout=60,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        owns_client = True
    try:
        for row in papers:
            document = _paper_document(db, row)
            document["source"] = source_scope
            if source_scope == "fulltext":
                assert config is not None and pdf_client is not None
                try:
                    _paper, text = _paper_text(
                        config, db, int(row["id"]), "fulltext", pdf_client
                    )
                except Exception as exc:
                    raise ValueError(f"Could not load PDF for {row['title']}: {exc}") from exc
                document["fulltext"] = text
            documents.append(document)
    finally:
        if owns_client and pdf_client is not None:
            pdf_client.close()
    return documents

def _query_prompt(user_prompt: str, documents: list[dict[str, Any]], *, merge: bool) -> str:
    if merge:
        instruction = (
            "Merge the partial outputs into one final result. Remove duplicates, preserve paper_id "
            "and exact titles, and do not introduce claims absent from the partial outputs."
        )
    else:
        instruction = (
            "Apply the user's prompt independently and comparatively across the supplied records. "
            "Every item must identify the relevant paper_id and exact title."
        )
    return f"""You are analyzing a filtered collection of research papers.
Return exactly one JSON object matching the supplied schema. {instruction}
The top-level summary should synthesize the collection. The items list should contain concrete, useful outputs rather than restating titles. Do not invent facts.

USER PROMPT:
{user_prompt}

RECORDS:
{json.dumps(documents, ensure_ascii=False)}
"""


def _generate(llm: LLMClient, prompt: str) -> tuple[BatchQueryResult, dict[str, Any]]:
    response = llm.generate_json(
        prompt,
        schema=BATCH_QUERY_JSON_SCHEMA,
        schema_name="paper_collection_query",
    )
    try:
        return BatchQueryResult.model_validate(response.data), response.usage
    except ValidationError as exc:
        raise LLMError(f"Invalid collection-query result: {exc}") from exc


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _compact_document(document: dict[str, Any], budget: int) -> dict[str, Any]:
    """Keep a query record useful while guaranteeing that it fits one group."""
    if _json_size([document]) <= budget:
        return document

    compact = {
        key: document[key]
        for key in ("paper_id", "title", "publication_date", "source")
        if document.get(key) is not None
    }
    source = next(
        (
            document[key]
            for key in ("abstract", "summary", "output", "items")
            if document.get(key) not in (None, "", [], {})
        ),
        document,
    )
    raw = source if isinstance(source, str) else json.dumps(source, ensure_ascii=False)
    marker = "\n[record truncated for collection query]"
    compact["content_excerpt"] = marker
    available = max(0, budget - _json_size([compact]) - 8)
    compact["content_excerpt"] = raw[:available] + (marker if len(raw) > available else "")

    # JSON escaping can make the serialized excerpt larger than its character count.
    while _json_size([compact]) > budget and compact["content_excerpt"]:
        excess = _json_size([compact]) - budget
        compact["content_excerpt"] = compact["content_excerpt"][: -max(1, excess)]

    if _json_size([compact]) <= budget:
        return compact

    # Pathological tiny budgets: retain identity before falling back to an empty record.
    for key in ("source", "publication_date", "title", "paper_id"):
        compact.pop(key, None)
        if _json_size([compact]) <= budget:
            return compact
    return {} if _json_size([{}]) <= budget else document


def _pack_documents(documents: list[dict[str, Any]], budget: int) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for document in documents:
        document = _compact_document(document, budget)
        candidate = [*current, document]
        if current and _json_size(candidate) > budget:
            groups.append(current)
            current = []
        current.append(document)
    if current:
        groups.append(current)
    return groups


def _split_oversized_document(document: dict[str, Any], budget: int) -> list[dict[str, Any]]:
    if _json_size([document]) <= budget:
        return [document]
    content_key = next(
        (key for key in ("fulltext", "abstract", "content_excerpt") if isinstance(document.get(key), str)),
        None,
    )
    if content_key is None:
        return [_compact_document(document, budget)]
    content = str(document[content_key])
    shell = {key: value for key, value in document.items() if key != content_key}
    shell["chunk_index"] = 1
    shell["chunk_count"] = 1
    room = max(400, budget - _json_size([shell]) - 200)
    chunks = [content[index:index + room] for index in range(0, len(content), room)] or [""]
    result: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, 1):
        item = dict(shell)
        item["chunk_index"] = index
        item["chunk_count"] = len(chunks)
        item[content_key] = chunk
        result.append(_compact_document(item, budget))
    return result


def run_query_for_model(
    llm: LLMClient,
    *,
    user_prompt: str,
    documents: list[dict[str, Any]],
    context_window_tokens: int,
    max_input_chars: int,
    pdf_strategy: str = "auto",
    progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[BatchQueryResult, dict[str, Any], int]:
    if not documents:
        return (
            BatchQueryResult(
                title="No matching papers",
                summary="No stored papers matched the selected filters.",
                items=[],
            ),
            {},
            0,
        )
    if pdf_strategy not in {"auto", "direct", "chunked"}:
        raise ValueError("PDF strategy must be auto, direct, or chunked")
    budget = context_char_budget(context_window_tokens, max_input_chars)
    record_budget = max(1_500, int(budget * 0.75))
    direct_fits = _json_size(documents) <= record_budget
    if pdf_strategy == "direct" and not direct_fits:
        raise LLMError(
            "The selected papers exceed the context window; use Auto or Chunk + merge, "
            "select fewer papers, or choose a larger context window"
        )
    if direct_fits and pdf_strategy in {"auto", "direct"}:
        groups = [documents]
    else:
        expanded = [
            chunk
            for document in documents
            for chunk in _split_oversized_document(document, record_budget)
        ]
        groups = _pack_documents(expanded, record_budget)
    usage: dict[str, Any] = {}
    partials: list[dict[str, Any]] = []
    input_chars = 0
    for group_index, group in enumerate(groups):
        check_cancelled(cancel_event)
        publish(progress, "message", f"Processing record group {group_index + 1}/{len(groups)}")
        prompt = _query_prompt(user_prompt, group, merge=False)
        input_chars += len(prompt)
        result, call_usage = _generate(llm, prompt)
        partials.append(result.model_dump())
        for key, value in call_usage.items():
            if isinstance(value, (int, float)):
                usage[key] = usage.get(key, 0) + value
            elif key not in usage:
                usage[key] = value

    for round_index in range(6):
        check_cancelled(cancel_event)
        if len(partials) == 1:
            return BatchQueryResult.model_validate(partials[0]), usage, input_chars
        merge_groups = _pack_documents(partials, record_budget)
        if len(merge_groups) >= len(partials):
            raise LLMError("Collection output exceeds the selected context window")
        next_partials: list[dict[str, Any]] = []
        for group_index, group in enumerate(merge_groups):
            check_cancelled(cancel_event)
            publish(progress, "message", f"Merging outputs {group_index + 1}/{len(merge_groups)} · round {round_index + 1}")
            prompt = _query_prompt(user_prompt, group, merge=True)
            input_chars += len(prompt)
            result, call_usage = _generate(llm, prompt)
            next_partials.append(result.model_dump())
            for key, value in call_usage.items():
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0) + value
                elif key not in usage:
                    usage[key] = value
        partials = next_partials
    raise LLMError("Could not merge collection-query outputs")

def run_query_batch(
    config_path: Path,
    batch_id: str,
    query_id: int,
    profile_names: tuple[str, ...],
    context_window_tokens: int,
    progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
    active_client: Callable[[LLMClient | None], None] | None = None,
) -> dict[str, int | bool]:
    config = load_config(config_path)
    profiles = {item.profile: item for item in config.available_llms}
    completed = 0
    failed = 0
    publish(progress, "stage", "Preparing saved prompt")
    with analysis_lock(config.paths.database):
        with Database(config.paths.database) as db:
            query = db.get_saved_query(query_id)
            if query is None:
                raise ValueError("Saved query not found")
            runs = {row["profile"]: row for row in db.query_runs_for_batch(batch_id)}
            first_run = next(iter(runs.values()), None)
            selection: dict[str, Any] = {}
            if first_run is not None:
                try:
                    decoded = json.loads(first_run["selection_json"] or "{}")
                    if isinstance(decoded, dict):
                        selection = decoded
                except (TypeError, json.JSONDecodeError):
                    selection = {}
            try:
                check_cancelled(cancel_event)
            except JobCancelled:
                for pending in runs.values():
                    if pending["status"] in {"queued", "running"}:
                        db.finish_query_run(
                            int(pending["id"]), matched_count=0, input_chars=0,
                            error="Cancelled by user",
                        )
                return {"completed": completed, "failed": failed, "cancelled": True}
            source_scope = str(selection.get("source_scope") or "abstract")
            pdf_strategy = str(selection.get("pdf_strategy") or "auto")
            documents = collect_query_documents(
                db,
                source_scope,
                selection=selection,
                config=config,
            )
            publish(progress, "message", f"Matched {len(documents)} record(s)")
            for profile_name in profile_names:
                row = runs.get(profile_name)
                if row is None:
                    continue
                run_id = int(row["id"])
                profile = profiles.get(profile_name)
                if profile is None:
                    db.finish_query_run(
                        run_id, matched_count=len(documents), input_chars=0,
                        error=f"Configured LLM profile {profile_name!r} is no longer available",
                    )
                    failed += 1
                    continue
                try:
                    check_cancelled(cancel_event)
                except JobCancelled:
                    for pending in runs.values():
                        if pending["status"] in {"queued", "running"}:
                            db.finish_query_run(
                                int(pending["id"]), matched_count=len(documents),
                                input_chars=0, error="Cancelled by user"
                            )
                    return {"completed": completed, "failed": failed, "cancelled": True}
                db.start_query_run(run_id)
                llm: LLMClient | None = None
                try:
                    publish(progress, "reset_output", "")
                    publish(progress, "stage", f"Running saved prompt with {profile.model}")
                    llm = create_llm(profile)
                    configure_job_runtime(llm, progress, cancel_event)
                    if active_client is not None:
                        active_client(llm)
                    configure_context_window(llm, context_window_tokens)
                    result, _usage, input_chars = run_query_for_model(
                        llm, user_prompt=query["prompt"], documents=documents,
                        context_window_tokens=context_window_tokens,
                        max_input_chars=profile.max_input_chars,
                        pdf_strategy=pdf_strategy,
                        progress=progress,
                        cancel_event=cancel_event,
                    )
                    check_cancelled(cancel_event)
                    db.finish_query_run(
                        run_id, matched_count=len(documents), input_chars=input_chars,
                        output=result.model_dump(),
                    )
                    completed += 1
                    publish(progress, "message", f"Completed {profile.model}")
                except JobCancelled:
                    db.finish_query_run(
                        run_id, matched_count=len(documents), input_chars=0,
                        error="Cancelled by user",
                    )
                    for pending in runs.values():
                        if int(pending["id"]) != run_id and pending["status"] == "queued":
                            db.finish_query_run(
                                int(pending["id"]), matched_count=len(documents),
                                input_chars=0, error="Cancelled by user",
                            )
                    return {"completed": completed, "failed": failed, "cancelled": True}
                except Exception as exc:
                    db.finish_query_run(
                        run_id, matched_count=len(documents), input_chars=0, error=str(exc)
                    )
                    failed += 1
                    publish(progress, "message", f"{profile.model} failed: {exc}")
                finally:
                    if active_client is not None:
                        active_client(None)
                    if llm is not None:
                        llm.close()
    publish(progress, "stage", "Complete")
    return {"completed": completed, "failed": failed}


class QueryController:
    def __init__(
        self, config_path: str | Path, runner: Callable[..., dict[str, int | bool]] | None = None
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self._runner = runner or run_query_batch
        self._lock = threading.Lock()
        self._snapshot = QueryBatchSnapshot()
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._active_llm: LLMClient | None = None

    def start(
        self,
        query_id: int,
        profiles: tuple[str, ...],
        context_window_tokens: int,
        *,
        selection: dict[str, Any] | None = None,
        source_scope: str = "abstract",
        pdf_strategy: str = "auto",
    ) -> str | None:
        with self._lock:
            if self._snapshot.running:
                return None
            if not 512 <= context_window_tokens <= 1_000_000:
                raise ValueError("Context window must be between 512 and 1000000 tokens")
            if source_scope not in {"abstract", "fulltext"}:
                raise ValueError("Query source must be abstract or fulltext")
            if pdf_strategy not in {"auto", "direct", "chunked"}:
                raise ValueError("PDF strategy must be auto, direct, or chunked")
            selection = dict(selection or {})
            selection["source_scope"] = source_scope
            selection["pdf_strategy"] = pdf_strategy
            config = load_config(self.config_path)
            available = {item.profile: item for item in config.available_llms}
            selected = tuple(dict.fromkeys(profiles))
            if not selected:
                raise ValueError("Select at least one configured model")
            unknown = [name for name in selected if name not in available]
            if unknown:
                raise ValueError(f"Unknown configured model profile(s): {', '.join(unknown)}")
            batch_id = secrets.token_hex(10)
            with Database(config.paths.database) as db:
                if db.get_saved_query(query_id) is None:
                    raise ValueError("Saved query not found")
                validate_query_sources(
                    db,
                    selection=selection,
                    source_scope=source_scope,
                    pdf_strategy=pdf_strategy,
                )
                db.create_query_runs(
                    batch_id=batch_id, query_id=query_id,
                    profiles=[(name, available[name].provider, available[name].model) for name in selected],
                    context_window_tokens=context_window_tokens, selection=selection,
                )
            paper_id = None
            if selection and isinstance(selection.get("paper_ids"), list) and selection["paper_ids"]:
                try:
                    paper_id = int(selection["paper_ids"][0])
                except (TypeError, ValueError):
                    paper_id = None
            self._cancel_event = threading.Event()
            self._active_llm = None
            self._snapshot = QueryBatchSnapshot(
                running=True, batch_id=batch_id, query_id=query_id, paper_id=paper_id,
                profiles=selected, stage="Queued", message="Preparing saved prompt",
            )
            self._thread = threading.Thread(
                target=self._work, args=(batch_id, query_id, selected, context_window_tokens),
                name="paperwatch-query-batch", daemon=True,
            )
            self._thread.start()
            return batch_id

    def _progress(self, kind: str, value: str) -> None:
        with self._lock:
            if kind == "stage":
                self._snapshot.stage = value[:200]
            elif kind == "message":
                self._snapshot.message = value[:1000]
            elif kind == "reset_output":
                self._snapshot.live_output = ""
            elif kind == "output":
                self._snapshot.live_output = (self._snapshot.live_output + value)[-100_000:]

    def _set_active_llm(self, llm: LLMClient | None) -> None:
        with self._lock:
            self._active_llm = llm

    def cancel(self) -> bool:
        with self._lock:
            if not self._snapshot.running:
                return False
            self._snapshot.cancel_requested = True
            self._snapshot.message = "Stopping after the current response boundary"
            self._cancel_event.set()
            llm = self._active_llm
        if llm is not None:
            try:
                llm.close()
            except Exception:
                pass
        return True

    def _work(self, batch_id: str, query_id: int, profiles: tuple[str, ...], context: int) -> None:
        try:
            parameter_count = len(inspect.signature(self._runner).parameters)
            if parameter_count <= 5:
                result = self._runner(self.config_path, batch_id, query_id, profiles, context)
            else:
                result = self._runner(
                    self.config_path, batch_id, query_id, profiles, context,
                    self._progress, self._cancel_event, self._set_active_llm,
                )
            with self._lock:
                self._snapshot.running = False
                self._snapshot.completed = int(result.get("completed", 0))
                self._snapshot.failed = int(result.get("failed", 0))
                self._snapshot.cancelled = bool(result.get("cancelled", False))
                self._snapshot.cancel_requested = False
                self._snapshot.stage = "Cancelled" if self._snapshot.cancelled else "Complete"
                if self._snapshot.cancelled:
                    self._snapshot.message = "Saved prompt cancelled"
        except Exception as exc:
            try:
                config = load_config(self.config_path)
                with Database(config.paths.database) as db:
                    for row in db.query_runs_for_batch(batch_id):
                        if row["status"] in {"queued", "running"}:
                            db.finish_query_run(
                                int(row["id"]), matched_count=0, input_chars=0, error=str(exc)
                            )
            finally:
                with self._lock:
                    self._snapshot.running = False
                    self._snapshot.error = str(exc)
                    self._snapshot.stage = "Failed"
                    self._snapshot.message = str(exc)
        finally:
            with self._lock:
                self._active_llm = None

    def snapshot(self) -> QueryBatchSnapshot:
        with self._lock:
            return QueryBatchSnapshot(**asdict(self._snapshot))

    def wait(self, timeout: float = 10) -> None:
        if self._thread:
            self._thread.join(timeout)
