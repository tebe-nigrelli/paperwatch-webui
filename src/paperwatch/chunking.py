from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from .analysis import AnalysisOutcome
from .config import LLMConfig
from .db import Database
from .job_runtime import ProgressCallback, check_cancelled, publish
from .llm import LLMClient, LLMError, PromptBuilder
from .models import (
    CHUNK_NOTES_JSON_SCHEMA,
    AnalysisResult,
    ChunkNotes,
    PaperRecord,
    model_attribution,
    stable_hash,
)


@dataclass(frozen=True, slots=True)
class ChunkedAnalysisOutcome:
    outcome: AnalysisOutcome
    chunk_count: int


def context_prompt_budget(context_window_tokens: int) -> int:
    """Conservative total prompt budget, reserving room for generated JSON."""
    if context_window_tokens < 512:
        raise ValueError("Context window must be at least 512 tokens")
    return int(context_window_tokens * 3.2 * 0.55)


def context_char_budget(context_window_tokens: int, max_input_chars: int) -> int:
    """Source-material budget capped by the configured input limit."""
    return min(max_input_chars, context_prompt_budget(context_window_tokens))


def direct_prompt_fit(
    *,
    prompt_path,
    max_input_chars: int,
    paper: PaperRecord,
    text: str,
    scope: str,
    model_name: str,
    context_window_tokens: int,
) -> tuple[bool, int, int]:
    """Return whether a complete strict prompt fits the selected context budget."""
    builder = PromptBuilder(prompt_path, max_input_chars)
    budget = context_prompt_budget(context_window_tokens)
    try:
        prompt = builder.build(paper, text, scope, model_attribution(model_name))
    except LLMError:
        return False, 0, budget
    return len(prompt) <= budget, len(prompt), budget


def split_text(text: str, chunk_chars: int, *, overlap_chars: int = 300) -> list[str]:
    if chunk_chars < 500:
        raise ValueError("Chunk size is too small")
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        raise ValueError("PDF text is empty")
    if len(normalized) <= chunk_chars:
        return [normalized]
    overlap = min(max(overlap_chars, 0), chunk_chars // 5)
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        target = min(len(normalized), start + chunk_chars)
        end = target
        if target < len(normalized):
            search_start = start + chunk_chars // 2
            candidates = [
                normalized.rfind("\n\n", search_start, target),
                normalized.rfind("\n", search_start, target),
                normalized.rfind(". ", search_start, target),
                normalized.rfind(" ", search_start, target),
            ]
            best = max(candidates)
            if best > start:
                end = best + (2 if normalized[best : best + 2] in {"\n\n", ". "} else 1)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        next_start = max(start + 1, end - overlap)
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


def _chunk_prompt(paper: PaperRecord, text: str, index: int, total: int) -> str:
    return f"""You are extracting rigorous notes from chunk {index + 1} of {total} of a research paper.
Return exactly one JSON object matching the supplied schema. Record only material actually present in this chunk. Do not infer missing sections or fabricate citations. Intermediate chunk notes are not subject to the final summary's 200-word field cap. Keep substantive_content below 800 words and each list item below 200 words while preserving all material needed for the final merge. Preserve mathematical notation with escaped LaTeX.

Paper: {paper.title}
Authors: {', '.join(author.name for author in paper.authors)}
Chunk: {index + 1}/{total}

--- BEGIN CHUNK ---
{text}
--- END CHUNK ---
"""


def _reduction_prompt(items: list[dict[str, Any]], round_index: int) -> str:
    return f"""Merge the following structured research-paper notes into one compact JSON object matching the supplied schema.
Preserve distinct definitions, assumptions, methods, results, evidence, limitations, and exact bibliography titles. Remove repetition. Do not add facts. The section field should be \"Consolidated notes, reduction {round_index}\".

{json.dumps(items, ensure_ascii=False)}
"""


def _sum_usage(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    for key, value in incoming.items():
        if isinstance(value, (int, float)) and isinstance(target.get(key, 0), (int, float)):
            target[key] = target.get(key, 0) + value
        elif key not in target:
            target[key] = value


def _generate_notes(
    llm: LLMClient,
    prompt: str,
) -> tuple[ChunkNotes, dict[str, Any]]:
    if not hasattr(llm, "generate_json"):
        raise LLMError("This model client does not support chunked structured generation")
    response = llm.generate_json(
        prompt,
        schema=CHUNK_NOTES_JSON_SCHEMA,
        schema_name="paper_chunk_notes",
    )
    try:
        return ChunkNotes.model_validate(response.data), response.usage
    except ValidationError as exc:
        raise LLMError(f"Invalid chunk summary: {exc}") from exc


def _reduce_to_budget(
    llm: LLMClient,
    notes: list[ChunkNotes],
    budget: int,
    usage: dict[str, Any],
    *,
    progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> str:
    current = [item.model_dump() for item in notes]
    for round_index in range(1, 7):
        check_cancelled(cancel_event)
        encoded = json.dumps(current, ensure_ascii=False)
        if len(encoded) <= budget:
            return encoded
        groups: list[list[dict[str, Any]]] = []
        group: list[dict[str, Any]] = []
        group_size = 2
        for item in current:
            size = len(json.dumps(item, ensure_ascii=False))
            if group and group_size + size > budget:
                groups.append(group)
                group = []
                group_size = 2
            group.append(item)
            group_size += size + 1
        if group:
            groups.append(group)
        if len(groups) >= len(current):
            # The model cannot receive even one note with this context setting.
            raise LLMError(
                "Chunk notes exceed the selected context window; choose a larger context window"
            )
        reduced: list[dict[str, Any]] = []
        for group_index, group_items in enumerate(groups):
            check_cancelled(cancel_event)
            publish(progress, "message", f"Reducing notes {group_index + 1}/{len(groups)} · round {round_index}")
            result, call_usage = _generate_notes(
                llm, _reduction_prompt(group_items, round_index)
            )
            _sum_usage(usage, call_usage)
            reduced.append(result.model_dump())
        current = reduced
    raise LLMError("Could not reduce chunk summaries within the selected context window")


def analyze_chunked_text(
    *,
    db: Database,
    prompt_path,
    llm_config: LLMConfig,
    llm: LLMClient,
    paper_id: int,
    paper: PaperRecord,
    text: str,
    context_window_tokens: int,
    progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> ChunkedAnalysisOutcome:
    check_cancelled(cancel_event)
    prompt_budget = context_prompt_budget(context_window_tokens)
    try:
        prompt_template_hash = stable_hash(prompt_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LLMError(f"Cannot read prompt template: {exc}") from exc
    cache_variant = llm_config.reasoning_effort or "default"
    final_digest = stable_hash(
        f"chunked:{context_window_tokens}:{cache_variant}:{stable_hash(text)}:{prompt_template_hash}"
    )
    cached_final = db.get_analysis(
        paper_id,
        "fulltext",
        llm.provider,
        llm.model,
        llm_config.prompt_version,
        final_digest,
    )
    if cached_final is not None:
        return ChunkedAnalysisOutcome(
            outcome=AnalysisOutcome(
                result=AnalysisResult.model_validate_json(cached_final["result_json"]),
                analysis_id=int(cached_final["id"]),
                created=False,
            ),
            chunk_count=int(cached_final["chunk_count"] or 0),
        )
    builder = PromptBuilder(prompt_path, llm_config.max_input_chars)
    final_prefix = (
        "The following structured notes were extracted from every chunk of the complete PDF. "
        "Synthesize them into the required final paper summary. Treat them as evidence from the "
        "paper, preserve disagreements or uncertainty, and do not infer facts absent from the notes.\n\n"
    )
    fixed_final_prompt = builder.build(
        paper, final_prefix, "fulltext_chunked", model_attribution(llm.model)
    )
    final_notes_budget = prompt_budget - len(fixed_final_prompt)
    if final_notes_budget < 500:
        raise LLMError(
            "The selected context window is too small for the strict final summary prompt; "
            "choose a larger context window"
        )

    chunk_overhead = len(_chunk_prompt(paper, "", 0, 1))
    chunk_source_budget = min(
        llm_config.max_input_chars, prompt_budget - chunk_overhead
    )
    if chunk_source_budget < 500:
        raise LLMError(
            "The selected context window is too small for chunk summarization; "
            "choose a larger context window"
        )
    chunks = split_text(text, max(500, int(chunk_source_budget * 0.9)))
    notes: list[ChunkNotes] = []
    usage: dict[str, Any] = {}
    for index, chunk in enumerate(chunks):
        check_cancelled(cancel_event)
        publish(progress, "message", f"Processing chunk {index + 1}/{len(chunks)}")
        digest = stable_hash(f"{cache_variant}:{chunk}")
        cached = db.get_analysis_chunk(
            paper_id=paper_id,
            provider=llm.provider,
            model=llm.model,
            prompt_version=llm_config.prompt_version,
            content_sha256=digest,
            chunk_index=index,
            context_window_tokens=context_window_tokens,
        )
        if cached is not None:
            notes.append(ChunkNotes.model_validate_json(cached["summary_json"]))
            continue
        result, call_usage = _generate_notes(
            llm, _chunk_prompt(paper, chunk, index, len(chunks))
        )
        _sum_usage(usage, call_usage)
        db.save_analysis_chunk(
            paper_id=paper_id,
            provider=llm.provider,
            model=llm.model,
            prompt_version=llm_config.prompt_version,
            content_sha256=digest,
            chunk_index=index,
            context_window_tokens=context_window_tokens,
            summary=result.model_dump(),
            usage=call_usage,
        )
        notes.append(result)

    publish(progress, "message", "Combining chunk summaries")
    consolidated = _reduce_to_budget(
        llm, notes, final_notes_budget, usage, progress=progress, cancel_event=cancel_event
    )
    source_text = final_prefix + consolidated
    prompt = builder.build(
        paper, source_text, "fulltext_chunked", model_attribution(llm.model)
    )
    if len(prompt) > prompt_budget:
        raise LLMError(
            "The reduced chunk notes still exceed the selected context window; "
            "choose a larger context window"
        )
    check_cancelled(cancel_event)
    publish(progress, "message", "Generating final merged summary")
    response = llm.analyze(prompt)
    _sum_usage(usage, response.usage)
    source = paper.doi or paper.arxiv_id or paper.openalex_id or response.result.source
    result = response.result.model_copy(
        update={
            "name": paper.title,
            "generated_by_model": model_attribution(llm.model),
            "authors": [author.name for author in paper.authors],
            "source": source,
        }
    )
    analysis_id = db.save_analysis(
        paper_id=paper_id,
        scope="fulltext",
        provider=llm.provider,
        model=llm.model,
        prompt_version=llm_config.prompt_version,
        content_sha256=final_digest,
        input_chars=len(text),
        result=result,
        usage=usage,
        analysis_strategy="chunked",
        context_window_tokens=context_window_tokens,
        chunk_count=len(chunks),
    )
    return ChunkedAnalysisOutcome(
        outcome=AnalysisOutcome(result=result, analysis_id=analysis_id, created=True),
        chunk_count=len(chunks),
    )
