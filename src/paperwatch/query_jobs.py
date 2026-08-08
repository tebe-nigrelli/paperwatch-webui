from __future__ import annotations

import inspect
import json
import bisect
import os
import re
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

import httpx
import secrets
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from . import USER_AGENT
from .analysis_jobs import _paper_pdf_path, _paper_text
from .chunking import context_char_budget
from .config import load_config
from .db import Database
from .job_runtime import JobCancelled, ProgressCallback, check_cancelled, publish
from .llm import (
    LLMClient, LLMError, configure_context_window, configure_job_runtime, create_llm
)
from .locking import analysis_lock
from .models import BATCH_QUERY_JSON_SCHEMA, BatchQueryResult


class QuerySelectionWarning(ValueError):
    """A query could not start because every matching record was incompatible."""


@dataclass(slots=True)
class QueryBatchSnapshot:
    running: bool = False
    batch_id: str | None = None
    query_id: int | None = None
    paper_id: int | None = None
    blog_entry_id: int | None = None
    content_type: str = "papers"
    profiles: tuple[str, ...] = ()
    completed: int = 0
    failed: int = 0
    cancelled: bool = False
    cancel_requested: bool = False
    stage: str = ""
    message: str = ""
    warning: str = ""
    live_output: str = ""
    error: str | None = None
    progress_current: int = 0
    progress_total: int = 0


def _paper_document(db: Database, row: Any) -> dict[str, Any]:
    return {
        "paper_id": int(row["id"]),
        "blog_entry_id": None,
        "content_type": "paper",
        "title": row["title"],
        "publication_date": row["publication_date"],
        "authors": [item["name"] for item in db.paper_authors(int(row["id"]))],
        "venue": row["venue"],
        "abstract": row["abstract"],
        "doi": row["doi"],
        "arxiv_id": row["arxiv_id"],
        "openalex_id": row["openalex_id"],
    }


def _blog_document(row: Any) -> dict[str, Any]:
    authors = [
        name.strip()
        for name in str(row["author_names"] or "").split(",")
        if name.strip()
    ]
    published = str(row["published_at"] or row["fetched_at"] or "")
    return {
        "paper_id": None,
        "blog_entry_id": int(row["id"]),
        "content_type": "blog",
        "title": row["title"],
        "publication_date": published[:10] or None,
        "authors": authors,
        "blog": row["blog_title"],
        "entry_kind": row["entry_kind"],
        "change_kind": row["change_kind"],
        "source_url": row["source_url"] or row["url"],
        "content": row["normalized_text"],
        "diff": row["diff_text"] or None,
    }


def _regexp_excerpts_compiled(
    text: str, expression: re.Pattern[str], radius: int
) -> tuple[str, int]:
    matches = list(islice(expression.finditer(text), 500))
    if not matches:
        return "", 0
    ranges: list[tuple[int, int]] = []
    if radius == 0:
        # Radius zero is the common filter/tag case. Avoid tokenizing the entire
        # document; expand only the matched spans to their surrounding words.
        for match in matches:
            start = match.start()
            end = match.end()
            while start > 0 and not text[start - 1].isspace():
                start -= 1
            while end < len(text) and not text[end].isspace():
                end += 1
            ranges.append((start, end))
    else:
        words = list(re.finditer(r"\S+", text))
        starts = [item.start() for item in words]
        ends = [item.end() for item in words]
        for match in matches:
            if not words:
                ranges.append((match.start(), match.end()))
                continue
            first = bisect.bisect_right(ends, match.start())
            last = max(
                first,
                bisect.bisect_left(starts, max(match.end(), match.start() + 1)),
            )
            left = max(0, first - radius)
            right = min(len(words), max(first + 1, last) + radius)
            start = words[left].start() if left < len(words) else match.start()
            end = words[right - 1].end() if right else match.end()
            ranges.append((start, end))
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    excerpts = [text[start:end].strip() for start, end in merged if text[start:end].strip()]
    return "\n\n[…]\n\n".join(excerpts), len(matches)


def _regexp_excerpts(text: str, pattern: str, radius: int) -> tuple[str, int]:
    return _regexp_excerpts_compiled(text, re.compile(pattern, re.MULTILINE), radius)


def _regexp_worker_count(count: int) -> int:
    if count <= 1:
        return 1
    return min(count, 32, max(2, (os.cpu_count() or 1) * 2))


def _regexp_document_match(
    document: dict[str, Any], expression: re.Pattern[str], radius: int
) -> tuple[dict[str, Any] | None, int]:
    content_key = next(
        (
            key
            for key in ("fulltext", "abstract", "content", "content_excerpt")
            if isinstance(document.get(key), str)
        ),
        None,
    )
    if content_key is None:
        return None, 0
    excerpt, matches = _regexp_excerpts_compiled(
        str(document[content_key]), expression, radius
    )
    if not matches:
        return None, 0
    item = dict(document)
    item[content_key] = excerpt
    item["regexp_match_count"] = matches
    return item, matches


def preprocess_regexp_documents(
    documents: list[dict[str, Any]], pattern: str, radius: int
) -> tuple[list[dict[str, Any]], int]:
    if not pattern:
        return documents, 0
    expression = re.compile(pattern, re.MULTILINE)
    workers = _regexp_worker_count(len(documents))
    if workers == 1:
        matches_by_document = [
            _regexp_document_match(document, expression, radius)
            for document in documents
        ]
    else:
        # Documents are independent; bounded threads avoid copying large full-text
        # strings into subprocesses and preserve input ordering.
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="paperwatch-regexp"
        ) as pool:
            matches_by_document = list(
                pool.map(
                    lambda document: _regexp_document_match(document, expression, radius),
                    documents,
                )
            )
    prepared = [item for item, _matches in matches_by_document if item is not None]
    total_matches = sum(matches for _item, matches in matches_by_document)
    return prepared, total_matches


def regexp_result(
    documents: list[dict[str, Any]], pattern: str, match_count: int
) -> BatchQueryResult:
    items = []
    for document in documents:
        content = next(
            (
                str(document[key])
                for key in ("fulltext", "abstract", "content", "content_excerpt")
                if document.get(key) not in (None, "")
            ),
            "",
        )
        items.append(
            {
                "paper_id": document.get("paper_id"),
                "blog_entry_id": document.get("blog_entry_id"),
                "content_type": document.get("content_type", "paper"),
                "title": str(document.get("title") or "Untitled record"),
                "output": content,
            }
        )
    return BatchQueryResult.model_validate(
        {
            "title": f"Regexp matches: {pattern}",
            "summary": f"{match_count} match(es) across {len(documents)} record(s).",
            "items": items,
        }
    )


def _regexp_tag_document(
    document: dict[str, Any], compiled: list[tuple[str, re.Pattern[str]]]
) -> tuple[dict[str, Any] | None, str | None]:
    content = next(
        (
            str(document[key])
            for key in ("fulltext", "abstract", "content", "content_excerpt")
            if document.get(key) not in (None, "")
        ),
        "",
    )
    for value, expression in compiled:
        match = expression.search(content)
        if match is None:
            continue
        return (
            {
                "paper_id": document.get("paper_id"),
                "blog_entry_id": document.get("blog_entry_id"),
                "content_type": document.get("content_type", "paper"),
                "title": str(document.get("title") or "Untitled record"),
                "output": match.group(0),
            },
            value,
        )
    return None, None


def regexp_tag_result(
    documents: list[dict[str, Any]], values: list[dict[str, str]]
) -> BatchQueryResult:
    compiled = [
        (item["value"], re.compile(item["pattern"], re.MULTILINE))
        for item in values
    ]
    workers = _regexp_worker_count(len(documents))
    if workers == 1:
        matches = [_regexp_tag_document(document, compiled) for document in documents]
    else:
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="paperwatch-regexp"
        ) as pool:
            matches = list(
                pool.map(
                    lambda document: _regexp_tag_document(document, compiled), documents
                )
            )
    items = [item for item, _value in matches if item is not None]
    counts: dict[str, int] = {}
    for _item, value in matches:
        if value is not None:
            counts[value] = counts.get(value, 0) + 1
    summary = ", ".join(f"{value}: {count}" for value, count in counts.items())
    return BatchQueryResult.model_validate(
        {
            "title": "Regexp tags",
            "summary": summary or "No regexp tags matched.",
            "items": items,
        }
    )


def _selection_search_args(selection: dict[str, Any] | None) -> dict[str, Any]:
    selection = selection or {}
    return {
        "query": str(selection.get("query") or ""),
        "title_query": str(selection.get("title_query") or ""),
        "author_id": str(selection.get("author_id") or ""),
        "author_ids": tuple(
            str(value)
            for value in selection.get("author_ids", ())
            if str(value).strip()
        ) if isinstance(selection.get("author_ids"), (list, tuple)) else (),
        "author_tags": tuple(
            str(tag)
            for tag in selection.get("author_tags", ())
            if str(tag).strip()
        ) if isinstance(selection.get("author_tags"), (list, tuple)) else (),
        "content_tags": tuple(
            str(tag)
            for tag in selection.get("content_tags", ())
            if str(tag).strip()
        ) if isinstance(selection.get("content_tags"), (list, tuple)) else (),
        "topic_id": str(selection.get("topic_id") or ""),
        "topic_ids": tuple(
            str(value)
            for value in selection.get("topic_ids", ())
            if str(value).strip()
        ) if isinstance(selection.get("topic_ids"), (list, tuple)) else (),
        "paper_kind": str(selection.get("paper_kind") or ""),
        "scope": str(selection.get("scope") or ""),
        "has_pdf": selection.get("has_pdf") if isinstance(selection.get("has_pdf"), bool) else None,
        "analyzed": selection.get("analyzed") if isinstance(selection.get("analyzed"), bool) else None,
        "starred": selection.get("starred") if isinstance(selection.get("starred"), bool) else None,
        "read": selection.get("read") if isinstance(selection.get("read"), bool) else None,
        "notes": selection.get("notes") if isinstance(selection.get("notes"), bool) else None,
        "date_from": str(selection.get("date_from") or ""),
        "date_to": str(selection.get("date_to") or ""),
        "sort": str(selection.get("sort") or "newest"),
    }


def _selected_papers(db: Database, selection: dict[str, Any] | None) -> list[Any]:
    selection = selection or {}
    selected_ids: list[int] = []
    raw_ids = (
        selection.get("resolved_paper_ids")
        if isinstance(selection.get("resolved_paper_ids"), list)
        else selection.get("paper_ids")
    )
    if isinstance(raw_ids, list):
        for value in raw_ids:
            try:
                selected_ids.append(int(value))
            except (TypeError, ValueError):
                continue
    excluded: set[int] = set()
    if isinstance(selection.get("excluded_paper_ids"), list):
        for value in selection["excluded_paper_ids"]:
            try:
                excluded.add(int(value))
            except (TypeError, ValueError):
                continue
    if selected_ids:
        return [
            row
            for paper_id in dict.fromkeys(selected_ids)
            if paper_id not in excluded and (row := db.get_paper(paper_id)) is not None
        ]
    papers, _ = db.search_papers(**_selection_search_args(selection), limit=None, offset=0)
    return [row for row in papers if int(row["id"]) not in excluded]


def _selection_stub_filter(selection: dict[str, Any] | None) -> bool | None:
    selection = selection or {}
    if "stub" in selection:
        value = selection.get("stub")
        if isinstance(value, bool) or value is None:
            return value
        normalized = str(value).strip().lower()
        if normalized in {"yes", "true", "1", "stub"}:
            return True
        if normalized in {"no", "false", "0", "full"}:
            return False
        if normalized in {"all", "any", "either", ""}:
            return None
    # Older saved selections used display_stubs=yes to mean no stub filtering.
    return None if bool(selection.get("display_stubs")) else False


def _selected_blog_entries(db: Database, selection: dict[str, Any] | None) -> list[Any]:
    selected_ids: list[int] = []
    raw_ids = (
        selection.get("resolved_blog_entry_ids")
        if selection and isinstance(selection.get("resolved_blog_entry_ids"), list)
        else selection.get("blog_entry_ids") if selection else None
    )
    if isinstance(raw_ids, list):
        for value in raw_ids:
            try:
                selected_ids.append(int(value))
            except (TypeError, ValueError):
                continue
        return db.blog_entries_by_ids(selected_ids, query_view=True) if selected_ids else []
    args = _selection_search_args(selection)
    return db.list_blog_entries(
        limit=None,
        query=str(args["query"]),
        title_query=str((selection or {}).get("title_query") or ""),
        tags=tuple(args["content_tags"]),
        author_id=str(args["author_id"]),
        author_ids=tuple(
            str(value)
            for value in (selection or {}).get("author_ids", ())
            if str(value).strip()
        ) if isinstance((selection or {}).get("author_ids"), (list, tuple)) else (),
        author_tags=tuple(args["author_tags"]),
        date_from=str(args["date_from"]),
        date_to=str(args["date_to"]),
        sort=str(args["sort"]),
        starred=args["starred"] if isinstance(args["starred"], bool) else None,
        read=args["read"] if isinstance(args["read"], bool) else None,
        analyzed=args["analyzed"] if isinstance(args["analyzed"], bool) else None,
        notes=args["notes"] if isinstance(args["notes"], bool) else None,
        stub=_selection_stub_filter(selection),
        query_view=True,
    )


def _paper_source_compatible(row: Any, source_scope: str) -> bool:
    abstract = bool(str(row["abstract"] or "").strip())
    pdf_url = bool(str(row["pdf_url"] or "").strip())
    local_value = str(row["local_pdf_path"] or "").strip()
    local_pdf = Path(local_value).expanduser() if local_value else None
    has_pdf = pdf_url or bool(local_pdf and local_pdf.is_file())
    if source_scope == "abstract":
        return abstract
    if source_scope in {"fulltext", "pdf"}:
        return has_pdf
    return abstract or has_pdf


def _append_selection_warning(selection: dict[str, Any], warning: str) -> None:
    warnings = selection.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
    if warning not in warnings:
        warnings.append(warning)
    selection["warnings"] = warnings


def _selection_warning_text(selection: dict[str, Any] | None) -> str:
    warnings = (selection or {}).get("warnings")
    if not isinstance(warnings, list):
        return ""
    return " ".join(str(item).strip() for item in warnings if str(item).strip())


def validate_query_sources(
    db: Database,
    *,
    selection: dict[str, Any] | None,
    source_scope: str,
    pdf_strategy: str,
) -> int:
    """Resolve compatible records, skipping individual papers that lack the requested source."""
    selection = selection if selection is not None else {}
    content_type = str(selection.get("content_type") or "papers")
    if content_type == "blogs":
        entries = _selected_blog_entries(db, selection)
        if not entries:
            raise ValueError("No blog entries match the current selection")
        if source_scope != "blog":
            raise ValueError("Blog saved prompts use the stored blog entry text")
        if pdf_strategy != "auto":
            raise ValueError("PDF strategy does not apply to blog entries")
        selection["resolved_blog_entry_ids"] = [int(row["id"]) for row in entries]
        return len(entries)
    if content_type != "papers":
        raise ValueError("Content type must be papers or blogs")
    papers = _selected_papers(db, selection)
    if not papers:
        raise ValueError("No papers match the current selection")
    if source_scope == "abstract" and pdf_strategy != "auto":
        raise ValueError("PDF strategy only applies when the source is full paper text")
    if source_scope == "pdf" and pdf_strategy != "auto":
        raise ValueError("PDF strategy does not apply when uploading the whole PDF")

    source_label = {
        "auto": "an abstract or PDF",
        "abstract": "an abstract",
        "fulltext": "a PDF",
        "pdf": "a PDF",
    }.get(source_scope, "a compatible source")
    compatible = [row for row in papers if _paper_source_compatible(row, source_scope)]
    missing = [row for row in papers if not _paper_source_compatible(row, source_scope)]
    if missing:
        examples = ", ".join(str(row["title"]) for row in missing[:3])
        remainder = len(missing) - min(3, len(missing))
        suffix = f" and {remainder} more" if remainder else ""
        noun = "paper" if len(missing) == 1 else "papers"
        warning = (
            f"Skipped {len(missing)} incompatible {noun} without {source_label}: "
            f"{examples}{suffix}."
        )
        _append_selection_warning(selection, warning)
        selection["excluded_paper_ids"] = [int(row["id"]) for row in missing]
    selection["resolved_paper_ids"] = [int(row["id"]) for row in compatible]
    if not compatible:
        warning = _selection_warning_text(selection) or "No compatible papers remain."
        raise QuerySelectionWarning(f"{warning} No compatible papers remain to run.")
    return len(compatible)


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
    content_type = str((selection or {}).get("content_type") or "papers")
    if content_type == "blogs":
        if source_scope not in {"blog", "papers"}:
            raise ValueError("Blog query source must be blog")
        return [_blog_document(row) for row in _selected_blog_entries(db, selection)]
    if content_type != "papers":
        raise ValueError("Content type must be papers or blogs")
    if source_scope == "papers":
        source_scope = "abstract"
    if source_scope not in {"abstract", "fulltext", "auto"}:
        raise ValueError(
            "Query source must be abstract or fulltext; automatic runs may use auto"
        )
    effective_selection = dict(selection or {})
    if search_query and not selection:
        effective_selection["query"] = search_query
    papers = _selected_papers(db, effective_selection)
    documents: list[dict[str, Any]] = []
    owns_client = False
    if source_scope in {"fulltext", "auto"} and pdf_client is None:
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
            effective_scope = source_scope
            if source_scope == "auto":
                local_value = str(row["local_pdf_path"] or "").strip()
                local_pdf = Path(local_value).expanduser() if local_value else None
                has_pdf = bool(str(row["pdf_url"] or "").strip()) or bool(
                    local_pdf and local_pdf.is_file()
                )
                effective_scope = "fulltext" if has_pdf else "abstract"
            document["source"] = effective_scope
            if effective_scope == "fulltext":
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
            "or blog_entry_id and exact titles, and do not introduce claims absent from the partial outputs."
        )
    else:
        instruction = (
            "Apply the user's prompt independently and comparatively across the supplied records. "
            "Every item must identify the relevant paper_id or blog_entry_id, content_type, and exact title."
        )
    return f"""You are analyzing a filtered collection of research papers and/or blog entries.
Return exactly one JSON object matching the supplied schema. {instruction}
The top-level summary should synthesize the collection.
The items list should contain concrete, useful outputs rather than restating titles.
Do not truncate requested output. Do not invent facts.

USER PROMPT:
{user_prompt}

RECORDS:
{json.dumps(documents, ensure_ascii=False)}
"""


QUERY_OUTPUT_SEGMENT_CHARS = 100_000


def _split_result_outputs(
    result: BatchQueryResult, segment_chars: int = QUERY_OUTPUT_SEGMENT_CHARS
) -> BatchQueryResult:
    """Split oversized item outputs into adjacent records without losing text."""
    if segment_chars < 1:
        raise ValueError("segment_chars must be positive")
    split_items = []
    changed = False
    for item in result.items:
        output = item.output
        if len(output) <= segment_chars:
            split_items.append(item)
            continue
        changed = True
        payload = item.model_dump()
        for start in range(0, len(output), segment_chars):
            part = dict(payload)
            part["output"] = output[start:start + segment_chars]
            split_items.append(type(item).model_validate(part))
    if not changed:
        return result
    return BatchQueryResult(
        title=result.title, summary=result.summary, items=split_items
    )


def _validated_query_result(data: Any) -> BatchQueryResult:
    return _split_result_outputs(BatchQueryResult.model_validate(data))


def _generate(llm: LLMClient, prompt: str) -> tuple[BatchQueryResult, dict[str, Any]]:
    response = llm.generate_json(
        prompt,
        schema=BATCH_QUERY_JSON_SCHEMA,
        schema_name="paper_collection_query",
    )
    try:
        return _validated_query_result(response.data), response.usage
    except ValidationError as exc:
        raise LLMError(f"Invalid collection-query result: {exc}") from exc


def _native_pdf_prompt(user_prompt: str, document: dict[str, Any]) -> str:
    metadata = {key: value for key, value in document.items() if key != "fulltext"}
    return f"""You are analyzing one complete research-paper PDF attached to this request.
Return exactly one JSON object matching the supplied schema. Apply the user's prompt to the
attached PDF, preserve the exact paper_id and title in every relevant item, and do not invent
facts. The top-level summary should be useful on its own.

USER PROMPT:
{user_prompt}

PAPER METADATA:
{json.dumps(metadata, ensure_ascii=False)}
"""


def run_query_for_native_pdfs(
    llm: LLMClient,
    *,
    user_prompt: str,
    papers: list[Any],
    db: Database,
    config: Any,
    context_window_tokens: int,
    max_input_chars: int,
    progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[BatchQueryResult, dict[str, Any], int]:
    generate_with_pdf = getattr(llm, "generate_json_with_pdf", None)
    if not callable(generate_with_pdf):
        raise LLMError(
            f"{getattr(llm, 'provider', 'Selected provider')} cannot read PDFs directly"
        )
    partials: list[dict[str, Any]] = []
    usage: dict[str, Any] = {}
    input_chars = 0
    with httpx.Client(
        timeout=60,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as pdf_client:
        for index, row in enumerate(papers, 1):
            check_cancelled(cancel_event)
            publish(progress, "message", f"Reading PDF {index}/{len(papers)} · {row['title']}")
            _paper, path = _paper_pdf_path(config, db, int(row["id"]), pdf_client)
            document = _paper_document(db, row)
            document["source"] = "fulltext"
            prompt = _native_pdf_prompt(user_prompt, document)
            input_chars += len(prompt)
            response = generate_with_pdf(
                prompt,
                path,
                schema=BATCH_QUERY_JSON_SCHEMA,
                schema_name="saved_query_result",
            )
            partials.append(_validated_query_result(response.data).model_dump())
            for key, value in response.usage.items():
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0) + value
                elif key not in usage:
                    usage[key] = value
    if len(partials) == 1:
        return _validated_query_result(partials[0]), usage, input_chars
    merged, merge_usage, merge_chars = run_query_for_model(
        llm,
        user_prompt="Merge these per-paper saved-query results into one collection result.",
        documents=partials,
        context_window_tokens=context_window_tokens,
        max_input_chars=max_input_chars,
        pdf_strategy="auto",
        content_type="papers",
        progress=progress,
        cancel_event=cancel_event,
    )
    input_chars += merge_chars
    for key, value in merge_usage.items():
        if isinstance(value, (int, float)):
            usage[key] = usage.get(key, 0) + value
        elif key not in usage:
            usage[key] = value
    return merged, usage, input_chars


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _text_chunks_for_budget(
    text: str,
    *,
    budget: int,
    build: Callable[[str], dict[str, Any]],
) -> list[str]:
    """Return exact consecutive slices whose wrapped JSON fits the budget."""
    if not text:
        if _json_size([build("")]) > budget:
            raise LLMError("Record metadata alone exceeds the selected context window")
        return [""]
    chunks: list[str] = []
    offset = 0
    while offset < len(text):
        lo, hi = 1, len(text) - offset
        best = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if _json_size([build(text[offset:offset + mid])]) <= budget:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best == 0:
            raise LLMError("Record metadata alone exceeds the selected context window")
        chunks.append(text[offset:offset + best])
        offset += best
    return chunks


def _split_batch_partial(document: dict[str, Any], budget: int) -> list[dict[str, Any]]:
    """Losslessly split an intermediate BatchQueryResult-shaped document."""
    title = str(document.get("title") or "Collection result")
    fragments: list[dict[str, Any]] = []
    summary = document.get("summary")
    if isinstance(summary, str) and summary:
        def summary_doc(chunk: str) -> dict[str, Any]:
            return {"title": title, "summary": chunk, "items": []}

        fragments.extend(
            summary_doc(chunk)
            for chunk in _text_chunks_for_budget(
                summary, budget=budget, build=summary_doc
            )
        )

    raw_items = document.get("items")
    if isinstance(raw_items, list):
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                encoded = json.dumps(raw_item, ensure_ascii=False, separators=(",", ":"))

                def raw_doc(chunk: str) -> dict[str, Any]:
                    return {"title": title, "summary": "", "item_fragment_json": chunk}

                fragments.extend(
                    raw_doc(chunk)
                    for chunk in _text_chunks_for_budget(
                        encoded, budget=budget, build=raw_doc
                    )
                )
                continue
            wrapper = {"title": title, "summary": "", "items": [raw_item]}
            if _json_size([wrapper]) <= budget:
                fragments.append(wrapper)
                continue
            output = raw_item.get("output")
            if isinstance(output, str):
                shell = {key: value for key, value in raw_item.items() if key != "output"}

                def item_doc(chunk: str) -> dict[str, Any]:
                    return {
                        "title": title,
                        "summary": "",
                        "items": [{**shell, "output": chunk}],
                    }

                fragments.extend(
                    item_doc(chunk)
                    for chunk in _text_chunks_for_budget(
                        output, budget=budget, build=item_doc
                    )
                )
                continue
            encoded = json.dumps(raw_item, ensure_ascii=False, separators=(",", ":"))

            def item_json_doc(chunk: str) -> dict[str, Any]:
                return {"title": title, "summary": "", "item_fragment_json": chunk}

            fragments.extend(
                item_json_doc(chunk)
                for chunk in _text_chunks_for_budget(
                    encoded, budget=budget, build=item_json_doc
                )
            )
    return fragments or [document]


def _split_oversized_document(document: dict[str, Any], budget: int) -> list[dict[str, Any]]:
    if _json_size([document]) <= budget:
        return [document]
    if isinstance(document.get("items"), list) and "summary" in document:
        fragments = _split_batch_partial(document, budget)
        if all(_json_size([fragment]) <= budget for fragment in fragments):
            return fragments

    content_key = next(
        (
            key
            for key in ("fulltext", "abstract", "content", "content_excerpt")
            if isinstance(document.get(key), str)
        ),
        None,
    )
    if content_key is not None:
        content = str(document[content_key])
        shell = {key: value for key, value in document.items() if key != content_key}
        # Reserve enough space for the final chunk counters before choosing slices.
        # This keeps every source character and preserves the existing chunk metadata.
        def sized_content_doc(chunk: str) -> dict[str, Any]:
            return {
                **shell,
                "chunk_index": 999999,
                "chunk_count": 999999,
                content_key: chunk,
            }

        chunks = _text_chunks_for_budget(
            content, budget=budget, build=sized_content_doc
        )
        count = len(chunks)
        return [
            {
                **shell,
                "chunk_index": index,
                "chunk_count": count,
                content_key: chunk,
            }
            for index, chunk in enumerate(chunks, 1)
        ]

    identity_keys = ("paper_id", "blog_entry_id", "content_type", "title")
    identity = {key: document[key] for key in identity_keys if document.get(key) is not None}
    payload = {key: value for key, value in document.items() if key not in identity}
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def fragment_doc(chunk: str) -> dict[str, Any]:
        return {**identity, "record_fragment_json": chunk}

    return [
        fragment_doc(chunk)
        for chunk in _text_chunks_for_budget(
            encoded, budget=budget, build=fragment_doc
        )
    ]


def _pack_documents(documents: list[dict[str, Any]], budget: int) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for document in documents:
        for fragment in _split_oversized_document(document, budget):
            if _json_size([fragment]) > budget:
                raise LLMError("A query fragment exceeds the selected context window")
            candidate = [*current, fragment]
            if current and _json_size(candidate) > budget:
                groups.append(current)
                current = []
            current.append(fragment)
    if current:
        groups.append(current)
    return groups


def _combine_partials_losslessly(partials: list[dict[str, Any]]) -> BatchQueryResult:
    """Fallback when another model merge cannot fit without dropping text."""
    results = [_validated_query_result(partial) for partial in partials]
    title = results[0].title if results else "Collection query result"
    summary = "\n\n".join(result.summary for result in results if result.summary)
    items = [item for result in results for item in result.items]
    return _split_result_outputs(
        BatchQueryResult(title=title, summary=summary, items=items)
    )


def run_query_for_model(
    llm: LLMClient,
    *,
    user_prompt: str,
    documents: list[dict[str, Any]],
    context_window_tokens: int,
    max_input_chars: int,
    pdf_strategy: str = "auto",
    content_type: str = "papers",
    progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[BatchQueryResult, dict[str, Any], int]:
    if content_type not in {"papers", "blogs"}:
        raise ValueError("Content type must be papers or blogs")
    if not documents:
        noun = "blog entries" if content_type == "blogs" else "papers"
        return (
            BatchQueryResult(
                title=f"No matching {noun}",
                summary=f"No stored {noun} matched the selected filters.",
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
            "The selected records exceed the context window; use Auto or Chunk + merge, "
            "select fewer records, or choose a larger context window"
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
            return _validated_query_result(partials[0]), usage, input_chars
        expanded_count = sum(
            len(_split_oversized_document(partial, record_budget))
            for partial in partials
        )
        merge_groups = _pack_documents(partials, record_budget)
        if len(merge_groups) >= expanded_count:
            # No context-safe merge can reduce these outputs. Preserve them all.
            return _combine_partials_losslessly(partials), usage, input_chars
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
    return _combine_partials_losslessly(partials), usage, input_chars

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
    progress_done = 0
    publish(progress, "stage", "Preparing saved prompt")
    with analysis_lock(config.paths.database):
        with Database(config.paths.database) as db:
            query = db.get_saved_query(query_id)
            if query is None:
                raise ValueError("Saved query not found")
            runs = {row["profile"]: row for row in db.query_runs_for_batch(batch_id)}
            publish(progress, "progress", f"0/{len(runs)}")
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
            content_type = str(selection.get("content_type") or "papers")
            regexp_pattern = str(query["regexp_pattern"] or "")
            regexp_radius = int(query["regexp_radius"] or 0)
            regexp_mode = str(query["regexp_mode"] or "query")
            query_kind = str(query["query_kind"] or "summary")
            label_values = Database._decoded_label_values(query["label_values_json"])
            native_profiles = {
                name
                for name in profile_names
                if name in profiles and profiles[name].provider == "claude-code"
            }
            native_pdf_only = (
                source_scope == "pdf"
                or (bool(profile_names) and native_profiles == set(profile_names))
            )
            native_papers: list[Any] = []
            if (
                native_pdf_only
                and content_type == "papers"
                and source_scope in {"fulltext", "pdf"}
                and not regexp_pattern
            ):
                native_papers = _selected_papers(db, selection)
                documents = [_paper_document(db, row) for row in native_papers]
            else:
                documents = collect_query_documents(
                    db,
                    source_scope,
                    selection=selection,
                    config=config,
                )
            regexp_matches = 0
            if regexp_pattern:
                documents, regexp_matches = preprocess_regexp_documents(
                    documents, regexp_pattern, regexp_radius
                )
                publish(
                    progress,
                    "message",
                    f"Regexp matched {regexp_matches} time(s) in {len(documents)} record(s)",
                )
            else:
                publish(progress, "message", f"Matched {len(documents)} record(s)")
            for profile_name in profile_names:
                row = runs.get(profile_name)
                if row is None:
                    continue
                run_id = int(row["id"])
                profile = profiles.get(profile_name)
                local_print = regexp_mode == "print" and bool(regexp_pattern)
                local_regexp_tags = query_kind == "regexp"
                if profile is None and not (local_print or local_regexp_tags):
                    db.finish_query_run(
                        run_id, matched_count=len(documents), input_chars=0,
                        error=f"Configured LLM profile {profile_name!r} is no longer available",
                    )
                    failed += 1
                    progress_done += 1
                    publish(progress, "progress", f"{progress_done}/{len(runs)}")
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
                    if local_regexp_tags:
                        result = regexp_tag_result(documents, label_values)
                        output = result.model_dump()
                        db.finish_query_run(
                            run_id,
                            matched_count=len(documents),
                            input_chars=sum(
                                len(str(item.get("content") or item.get("abstract") or item.get("fulltext") or ""))
                                for item in documents
                            ),
                            output=output,
                        )
                        db.apply_query_classifications(query_id, run_id, output)
                        completed += 1
                        progress_done += 1
                        publish(progress, "message", "Applied regexp tags")
                        publish(progress, "progress", f"{progress_done}/{len(runs)}")
                        continue
                    if local_print:
                        result = regexp_result(documents, regexp_pattern, regexp_matches)
                        db.finish_query_run(
                            run_id,
                            matched_count=len(documents),
                            input_chars=sum(
                                len(str(item.get("content") or item.get("abstract") or item.get("fulltext") or ""))
                                for item in documents
                            ),
                            output=result.model_dump(),
                        )
                        completed += 1
                        progress_done += 1
                        publish(progress, "message", "Printed regexp matches")
                        publish(progress, "progress", f"{progress_done}/{len(runs)}")
                        continue
                    assert profile is not None
                    publish(progress, "stage", f"Running saved prompt with {profile.model}")
                    llm = create_llm(profile)
                    configure_job_runtime(llm, progress, cancel_event)
                    if active_client is not None:
                        active_client(llm)
                    effective_context = (
                        profile.context_window_tokens
                        if profile.provider in {"codex", "claude-code"}
                        else context_window_tokens
                    )
                    configure_context_window(llm, effective_context)
                    native_pdf = (
                        profile.provider == "codex" and source_scope == "pdf"
                    ) or (
                        profile.provider == "claude-code" and source_scope == "fulltext"
                    )
                    if native_pdf and content_type == "papers" and not regexp_pattern:
                        if not native_papers:
                            native_papers = _selected_papers(db, selection)
                        result, _usage, input_chars = run_query_for_native_pdfs(
                            llm,
                            user_prompt=query["prompt"],
                            papers=native_papers,
                            db=db,
                            config=config,
                            context_window_tokens=profile.context_window_tokens,
                            max_input_chars=profile.max_input_chars,
                            progress=progress,
                            cancel_event=cancel_event,
                        )
                    else:
                        result, _usage, input_chars = run_query_for_model(
                            llm, user_prompt=query["prompt"], documents=documents,
                            context_window_tokens=effective_context,
                            max_input_chars=profile.max_input_chars,
                            pdf_strategy=pdf_strategy,
                            content_type=content_type,
                            progress=progress,
                            cancel_event=cancel_event,
                        )
                    check_cancelled(cancel_event)
                    output = result.model_dump()
                    db.finish_query_run(
                        run_id, matched_count=len(documents), input_chars=input_chars,
                        output=output,
                    )
                    db.apply_query_classifications(query_id, run_id, output)
                    completed += 1
                    publish(progress, "message", f"Completed {profile.model}")
                    progress_done += 1
                    publish(progress, "progress", f"{progress_done}/{len(runs)}")
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
                    model_name = profile.model if profile is not None else "regexp"
                    publish(progress, "message", f"{model_name} failed: {exc}")
                    progress_done += 1
                    publish(progress, "progress", f"{progress_done}/{len(runs)}")
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
        content_type: str | None = None,
    ) -> str | None:
        with self._lock:
            if self._snapshot.running:
                return None
            if not 512 <= context_window_tokens <= 1_000_000:
                raise ValueError("Context window must be between 512 and 1000000 tokens")
            selection = dict(selection or {})
            content_type = str(content_type or selection.get("content_type") or "papers")
            if content_type not in {"papers", "blogs"}:
                raise ValueError("Content type must be papers or blogs")
            if content_type == "blogs":
                source_scope = "blog"
                pdf_strategy = "auto"
                selection["content_type"] = "blogs"
            else:
                selection.pop("content_type", None)
                if source_scope not in {"abstract", "fulltext", "auto", "pdf"}:
                    raise ValueError(
                        "Query source must be abstract or fulltext, or whole-PDF upload; automatic runs may use auto"
                    )
            if pdf_strategy not in {"auto", "direct", "chunked"}:
                raise ValueError("PDF strategy must be auto, direct, or chunked")
            selection["source_scope"] = source_scope
            selection["pdf_strategy"] = pdf_strategy
            config = load_config(self.config_path)
            available = {item.profile: item for item in config.available_llms}
            batch_id = secrets.token_hex(10)
            with Database(config.paths.database) as db:
                query = db.get_saved_query(query_id)
                if query is None:
                    raise ValueError("Saved query not found")
                supported = str(query["content_scope"] or "both")
                if supported not in {"both", content_type}:
                    label = "papers" if supported == "papers" else "blogs"
                    raise ValueError(f"This query is configured for {label} only")
                local_print = (
                    str(query["regexp_mode"] or "query") == "print"
                    and bool(str(query["regexp_pattern"] or "").strip())
                )
                local_regexp_tags = str(query["query_kind"] or "summary") == "regexp"
                selected = (
                    ("regexp",)
                    if local_print or local_regexp_tags
                    else tuple(dict.fromkeys(profiles))
                )
                if not selected:
                    raise ValueError("Select at least one configured model")
                unknown = [name for name in selected if name not in available and name != "regexp"]
                if unknown:
                    raise ValueError(
                        f"Unknown configured model profile(s): {', '.join(unknown)}"
                    )
                configured_profiles = [available[name] for name in selected if name in available]
                if (
                    len(configured_profiles) == 1
                    and configured_profiles[0].provider in {"codex", "claude-code"}
                ):
                    context_window_tokens = configured_profiles[0].context_window_tokens
                if source_scope == "pdf":
                    if local_print or local_regexp_tags:
                        raise ValueError("Local regexp queries cannot upload a whole PDF")
                    if len(configured_profiles) != 1 or configured_profiles[0].provider != "codex":
                        raise ValueError("Whole-PDF upload requires exactly one Codex model")
                    if str(query["regexp_pattern"] or "").strip():
                        raise ValueError("Regexp preprocessing is unavailable for whole-PDF upload")
                    context_window_tokens = configured_profiles[0].context_window_tokens
                validate_query_sources(
                    db,
                    selection=selection,
                    source_scope=source_scope,
                    pdf_strategy=pdf_strategy,
                )
                warning = _selection_warning_text(selection)
                db.create_query_runs(
                    batch_id=batch_id, query_id=query_id,
                    profiles=[
                        (name, "local", "regexp")
                        if name == "regexp"
                        else (name, available[name].provider, available[name].model)
                        for name in selected
                    ],
                    context_window_tokens=context_window_tokens, selection=selection,
                )
            paper_id = None
            blog_entry_id = None
            if isinstance(selection.get("paper_ids"), list) and selection["paper_ids"]:
                try:
                    paper_id = int(selection["paper_ids"][0])
                except (TypeError, ValueError):
                    paper_id = None
            if isinstance(selection.get("blog_entry_ids"), list) and selection["blog_entry_ids"]:
                try:
                    blog_entry_id = int(selection["blog_entry_ids"][0])
                except (TypeError, ValueError):
                    blog_entry_id = None
            self._cancel_event = threading.Event()
            self._active_llm = None
            self._snapshot = QueryBatchSnapshot(
                running=True, batch_id=batch_id, query_id=query_id, paper_id=paper_id,
                blog_entry_id=blog_entry_id, content_type=content_type,
                profiles=selected, stage="Queued", message="Preparing saved prompt",
                warning=warning,
            )
            self._thread = threading.Thread(
                target=self._work, args=(batch_id, query_id, selected, context_window_tokens),
                name="paperwatch-query-batch", daemon=True,
            )
            self._thread.start()
            return batch_id

    def start_on_results(
        self,
        query_id: int,
        profiles: tuple[str, ...],
        context_window_tokens: int,
        *,
        targets: tuple[tuple[str, dict[str, Any]], ...],
        source_scope: str = "abstract",
        pdf_strategy: str = "auto",
    ) -> tuple[str, ...] | None:
        """Run one saved query over every supported result type in sequence."""
        with self._lock:
            if self._snapshot.running:
                return None
            if not 512 <= context_window_tokens <= 1_000_000:
                raise ValueError("Context window must be between 512 and 1000000 tokens")
            ordered_targets: list[tuple[str, dict[str, Any]]] = []
            seen_targets: set[str] = set()
            for content_type, raw_selection in targets:
                content_type = str(content_type)
                if content_type not in {"papers", "blogs"}:
                    raise ValueError("Content type must be papers or blogs")
                if content_type not in seen_targets:
                    ordered_targets.append((content_type, dict(raw_selection)))
                    seen_targets.add(content_type)
            if not ordered_targets:
                raise ValueError("No search results were selected")
            if "papers" in seen_targets and source_scope not in {
                "abstract", "fulltext", "auto", "pdf"
            }:
                raise ValueError(
                    "Query source must be abstract or fulltext, or whole-PDF upload; automatic runs may use auto"
                )
            if pdf_strategy not in {"auto", "direct", "chunked"}:
                raise ValueError("PDF strategy must be auto, direct, or chunked")

            config = load_config(self.config_path)
            available = {item.profile: item for item in config.available_llms}
            jobs: list[tuple[str, int, tuple[str, ...]]] = []
            active_types: list[str] = []
            warnings: list[str] = []
            with Database(config.paths.database) as db:
                query = db.get_saved_query(query_id)
                if query is None:
                    raise ValueError("Saved query not found")
                supported = str(query["content_scope"] or "both")
                local_print = (
                    str(query["regexp_mode"] or "query") == "print"
                    and bool(str(query["regexp_pattern"] or "").strip())
                )
                local_regexp_tags = str(query["query_kind"] or "summary") == "regexp"
                selected = (
                    ("regexp",)
                    if local_print or local_regexp_tags
                    else tuple(dict.fromkeys(profiles))
                )
                if not selected:
                    raise ValueError("Select at least one configured model")
                unknown = [name for name in selected if name not in available and name != "regexp"]
                if unknown:
                    raise ValueError(
                        f"Unknown configured model profile(s): {', '.join(unknown)}"
                    )
                configured_profiles = [available[name] for name in selected if name in available]
                if (
                    len(configured_profiles) == 1
                    and configured_profiles[0].provider in {"codex", "claude-code"}
                ):
                    context_window_tokens = configured_profiles[0].context_window_tokens
                if "papers" in seen_targets and source_scope == "pdf":
                    if local_print or local_regexp_tags:
                        raise ValueError("Local regexp queries cannot upload a whole PDF")
                    if len(configured_profiles) != 1 or configured_profiles[0].provider != "codex":
                        raise ValueError("Whole-PDF upload requires exactly one Codex model")
                    if str(query["regexp_pattern"] or "").strip():
                        raise ValueError("Regexp preprocessing is unavailable for whole-PDF upload")
                    context_window_tokens = configured_profiles[0].context_window_tokens

                for content_type, raw_selection in ordered_targets:
                    if supported not in {"both", content_type}:
                        continue
                    selection = dict(raw_selection)
                    if content_type == "blogs":
                        selection["content_type"] = "blogs"
                        target_source = "blog"
                        target_pdf_strategy = "auto"
                        if not _selected_blog_entries(db, selection):
                            continue
                    else:
                        selection.pop("content_type", None)
                        target_source = source_scope
                        target_pdf_strategy = pdf_strategy
                        if not _selected_papers(db, selection):
                            continue
                    selection["source_scope"] = target_source
                    selection["pdf_strategy"] = target_pdf_strategy
                    try:
                        validate_query_sources(
                            db,
                            selection=selection,
                            source_scope=target_source,
                            pdf_strategy=target_pdf_strategy,
                        )
                    except QuerySelectionWarning as exc:
                        warnings.append(str(exc))
                        continue
                    warning = _selection_warning_text(selection)
                    if warning:
                        warnings.append(warning)
                    batch_id = secrets.token_hex(10)
                    db.create_query_runs(
                        batch_id=batch_id,
                        query_id=query_id,
                        profiles=[
                            (name, "local", "regexp")
                            if name == "regexp"
                            else (name, available[name].provider, available[name].model)
                            for name in selected
                        ],
                        context_window_tokens=context_window_tokens,
                        selection=selection,
                    )
                    jobs.append((batch_id, query_id, selected))
                    active_types.append(content_type)
            if not jobs:
                if warnings:
                    raise QuerySelectionWarning(" ".join(dict.fromkeys(warnings)))
                raise ValueError("No records match the current search results")

            self._cancel_event = threading.Event()
            self._active_llm = None
            self._snapshot = QueryBatchSnapshot(
                running=True,
                batch_id=jobs[0][0],
                query_id=query_id,
                content_type="results" if len(jobs) > 1 else active_types[0],
                profiles=jobs[0][2],
                stage="Queued",
                message=f"Preparing saved prompt for {len(jobs)} result set(s)",
                warning=" ".join(dict.fromkeys(warnings)),
                progress_total=len(jobs),
            )
            self._thread = threading.Thread(
                target=self._work_many,
                args=(tuple(jobs), context_window_tokens),
                name="paperwatch-query-results",
                daemon=True,
            )
            self._thread.start()
            return tuple(job[0] for job in jobs)

    def start_many(
        self,
        query_ids: tuple[int, ...],
        profile: str,
        context_window_tokens: int,
        *,
        selection: dict[str, Any] | None = None,
        content_type: str = "blogs",
    ) -> tuple[str, ...] | None:
        """Queue several saved queries as one sequential collection job.

        This is intentionally single-model: it keeps the content-page runner
        predictable while still allowing several prompts to reuse the same
        filtered result set.
        """
        with self._lock:
            if self._snapshot.running:
                return None
            if not 512 <= context_window_tokens <= 1_000_000:
                raise ValueError("Context window must be between 512 and 1000000 tokens")
            ordered_ids = tuple(dict.fromkeys(int(value) for value in query_ids if int(value) > 0))
            if not ordered_ids:
                raise ValueError("Select at least one saved query")
            if content_type != "blogs":
                raise ValueError("Multi-query runs currently support blog results")
            selection = dict(selection or {})
            selection["content_type"] = "blogs"
            selection["source_scope"] = "blog"
            selection["pdf_strategy"] = "auto"
            config = load_config(self.config_path)
            available = {item.profile: item for item in config.available_llms}
            jobs: list[tuple[str, int, tuple[str, ...]]] = []
            with Database(config.paths.database) as db:
                validate_query_sources(
                    db,
                    selection=selection,
                    source_scope="blog",
                    pdf_strategy="auto",
                )
                warning = _selection_warning_text(selection)
                for query_id in ordered_ids:
                    query = db.get_saved_query(query_id)
                    if query is None:
                        raise ValueError(f"Saved query {query_id} was not found")
                    supported = str(query["content_scope"] or "both")
                    if supported not in {"both", "blogs"}:
                        raise ValueError(f"{query['name']} is configured for papers only")
                    local_print = (
                        str(query["regexp_mode"] or "query") == "print"
                        and bool(str(query["regexp_pattern"] or "").strip())
                    )
                    local_regexp_tags = str(query["query_kind"] or "summary") == "regexp"
                    selected = ("regexp",) if local_print or local_regexp_tags else (profile,)
                    if selected != ("regexp",) and profile not in available:
                        raise ValueError(f"Unknown configured model profile: {profile}")
                    batch_id = secrets.token_hex(10)
                    db.create_query_runs(
                        batch_id=batch_id,
                        query_id=query_id,
                        profiles=[
                            ("regexp", "local", "regexp")
                            if selected == ("regexp",)
                            else (profile, available[profile].provider, available[profile].model)
                        ],
                        context_window_tokens=context_window_tokens,
                        selection=selection,
                    )
                    jobs.append((batch_id, query_id, selected))
            self._cancel_event = threading.Event()
            self._active_llm = None
            self._snapshot = QueryBatchSnapshot(
                running=True,
                batch_id=jobs[0][0],
                query_id=jobs[0][1],
                content_type="blogs",
                profiles=(profile,),
                stage="Queued",
                message=f"Preparing {len(jobs)} saved queries",
                warning=warning,
                progress_total=len(jobs),
            )
            self._thread = threading.Thread(
                target=self._work_many,
                args=(tuple(jobs), context_window_tokens),
                name="paperwatch-query-sequence",
                daemon=True,
            )
            self._thread.start()
            return tuple(job[0] for job in jobs)

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
            elif kind == "progress":
                current, separator, total = value.partition("/")
                if separator:
                    try:
                        self._snapshot.progress_current = int(current)
                        self._snapshot.progress_total = int(total)
                    except ValueError:
                        pass

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

    def _work_many(
        self,
        jobs: tuple[tuple[str, int, tuple[str, ...]], ...],
        context: int,
    ) -> None:
        completed = 0
        failed = 0
        try:
            for index, (batch_id, query_id, profiles) in enumerate(jobs, 1):
                check_cancelled(self._cancel_event)
                with self._lock:
                    self._snapshot.batch_id = batch_id
                    self._snapshot.query_id = query_id
                    self._snapshot.stage = f"Query {index}/{len(jobs)}"
                    self._snapshot.message = "Preparing saved prompt"
                    self._snapshot.progress_current = index - 1
                    self._snapshot.progress_total = len(jobs)
                parameter_count = len(inspect.signature(self._runner).parameters)
                if parameter_count <= 5:
                    result = self._runner(self.config_path, batch_id, query_id, profiles, context)
                else:
                    result = self._runner(
                        self.config_path,
                        batch_id,
                        query_id,
                        profiles,
                        context,
                        self._progress,
                        self._cancel_event,
                        self._set_active_llm,
                    )
                completed += int(result.get("completed", 0))
                failed += int(result.get("failed", 0))
                with self._lock:
                    self._snapshot.progress_current = index
                    self._snapshot.progress_total = len(jobs)
            with self._lock:
                self._snapshot.running = False
                self._snapshot.completed = completed
                self._snapshot.failed = failed
                self._snapshot.cancel_requested = False
                self._snapshot.stage = "Complete"
                self._snapshot.message = f"Completed {len(jobs)} saved queries"
        except JobCancelled:
            with self._lock:
                self._snapshot.running = False
                self._snapshot.completed = completed
                self._snapshot.failed = failed
                self._snapshot.cancelled = True
                self._snapshot.cancel_requested = False
                self._snapshot.stage = "Cancelled"
                self._snapshot.message = "Saved queries cancelled"
        except Exception as exc:
            with self._lock:
                self._snapshot.running = False
                self._snapshot.completed = completed
                self._snapshot.failed = failed
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
