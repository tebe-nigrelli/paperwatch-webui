from __future__ import annotations

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
    digest = stable_hash(prompt)
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
