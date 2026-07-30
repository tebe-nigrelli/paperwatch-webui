from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import LLMConfig
from .db import Database
from .llm import LLMClient, PromptBuilder
from .models import AnalysisResult, PaperRecord, model_attribution, stable_hash


@dataclass(frozen=True, slots=True)
class AnalysisOutcome:
    result: AnalysisResult
    analysis_id: int
    created: bool


def analyze_text(
    *,
    db: Database,
    prompt_path: Path,
    llm_config: LLMConfig,
    llm: LLMClient,
    paper_id: int,
    paper: PaperRecord,
    text: str,
    scope: str,
) -> AnalysisOutcome:
    builder = PromptBuilder(prompt_path, llm_config.max_input_chars)
    prompt = builder.build(paper, text, scope, model_attribution(llm.model))
    digest = stable_hash(
        f"{prompt}\nprovider_settings:reasoning_effort={llm_config.reasoning_effort or 'default'}"
    )
    cached = db.get_analysis(
        paper_id,
        scope,
        llm.provider,
        llm.model,
        llm_config.prompt_version,
        digest,
    )
    if cached is not None:
        return AnalysisOutcome(
            result=AnalysisResult.model_validate(json.loads(cached["result_json"])),
            analysis_id=int(cached["id"]),
            created=False,
        )

    response = llm.analyze(prompt)
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
        scope=scope,
        provider=llm.provider,
        model=llm.model,
        prompt_version=llm_config.prompt_version,
        content_sha256=digest,
        input_chars=len(text),
        result=result,
        usage=response.usage,
    )
    return AnalysisOutcome(result=result, analysis_id=analysis_id, created=True)

def analyze_pdf_file(
    *,
    db: Database,
    prompt_path: Path,
    llm_config: LLMConfig,
    llm: LLMClient,
    paper_id: int,
    paper: PaperRecord,
    pdf_path: Path,
) -> AnalysisOutcome:
    """Analyze the original PDF as an attached file instead of extracted text."""
    path = Path(pdf_path)
    if not path.is_file():
        raise ValueError(f"PDF file does not exist: {path}")
    builder = PromptBuilder(prompt_path, llm_config.max_input_chars)
    attachment_note = (
        "The complete research paper is attached as a PDF file. Read the attached PDF in full, "
        "including figures, tables, appendices, footnotes, and bibliography. Do not rely on an "
        "extracted-text substitute."
    )
    prompt = builder.build(
        paper, attachment_note, "fulltext_pdf", model_attribution(llm.model)
    )
    digest_builder = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest_builder.update(block)
    file_hash = digest_builder.hexdigest()
    digest = stable_hash(
        f"{prompt}\nfile_sha256:{file_hash}\n"
        f"provider_settings:reasoning_effort={llm_config.reasoning_effort or 'default'}"
    )
    cached = db.get_analysis(
        paper_id,
        "fulltext",
        llm.provider,
        llm.model,
        llm_config.prompt_version,
        digest,
    )
    if cached is not None:
        return AnalysisOutcome(
            result=AnalysisResult.model_validate(json.loads(cached["result_json"])),
            analysis_id=int(cached["id"]),
            created=False,
        )
    analyze_pdf = getattr(llm, "analyze_pdf", None)
    if not callable(analyze_pdf):
        raise ValueError("The selected model provider does not support direct PDF upload")
    response = analyze_pdf(prompt, path)
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
        content_sha256=digest,
        input_chars=len(prompt),
        analysis_strategy="pdf_upload",
        result=result,
        usage=response.usage,
    )
    return AnalysisOutcome(result=result, analysis_id=analysis_id, created=True)
