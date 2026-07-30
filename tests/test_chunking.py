from __future__ import annotations

from dataclasses import replace

import pytest

from paperwatch.chunking import (
    _reduce_to_budget,
    analyze_chunked_text,
    context_char_budget,
    split_text,
)
from paperwatch.config import load_config
from paperwatch.db import Database
from paperwatch.llm import LLMError, StructuredJSONResponse
from paperwatch.models import ChunkNotes, LLMResponse


class FakeChunkLLM:
    provider = "ollama"
    model = "chunk-model"

    def __init__(self, analysis_result, *, invalid=False):
        self.analysis_result = analysis_result
        self.invalid = invalid
        self.chunk_calls: list[str] = []
        self.analysis_calls: list[str] = []

    def generate_json(self, prompt, *, schema, schema_name):
        self.chunk_calls.append(prompt)
        if self.invalid:
            return StructuredJSONResponse(data={"bad": True}, usage={}, raw={})
        reduction = "Merge the following" in prompt
        label = "Consolidated notes" if reduction else "Chunk notes"
        note = ChunkNotes(
            section=label,
            substantive_content=(
                "The chunk defines the setting, describes a compact method, and reports a bounded "
                "result under the stated evaluation conditions."
            ),
            definitions=["The reliability score measures stable predictions."],
            methods=["A compact agent transforms inputs into predictions."],
            results=["The method improves the reported score in the tested setting."],
            assumptions=["Inputs follow the benchmark protocol."],
            evidence=["The paper reports a benchmark comparison."],
            limitations=["The evaluation is narrow."],
            bibliography_titles=["A Prior Method"],
        )
        return StructuredJSONResponse(
            data=note.model_dump(), usage={"eval_count": 2}, raw={"ok": True}
        )

    def analyze(self, prompt):
        self.analysis_calls.append(prompt)
        return LLMResponse(result=self.analysis_result, usage={"eval_count": 3})

    def close(self):
        pass


def test_context_budget_and_split_boundaries():
    assert context_char_budget(8192, 100_000) > 10_000
    assert context_char_budget(8192, 2_000) == 2_000
    with pytest.raises(ValueError, match="512"):
        context_char_budget(100, 1_000)
    assert split_text("short text", 500) == ["short text"]
    chunks = split_text(("Paragraph one. " * 80) + "\n\n" + ("Paragraph two. " * 80), 600)
    assert len(chunks) > 2
    assert all(chunks)
    with pytest.raises(ValueError, match="too small"):
        split_text("text", 10)
    with pytest.raises(ValueError, match="empty"):
        split_text("   ", 500)


def test_chunked_analysis_caches_chunks_and_final_result(
    config_path, paper_record, analysis_result
):
    config = load_config(config_path)
    profile = replace(config.llm, model="chunk-model", max_input_chars=5000)
    text = ("Method and theorem details. " * 300).strip()
    llm = FakeChunkLLM(analysis_result)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        first = analyze_chunked_text(
            db=db,
            prompt_path=config.paths.prompt,
            llm_config=profile,
            llm=llm,
            paper_id=paper_id,
            paper=paper_record,
            text=text,
            context_window_tokens=1024,
        )
        assert first.outcome.created
        assert first.chunk_count > 1
        row = db.get_analysis_by_id(first.outcome.analysis_id)
        assert row["analysis_strategy"] == "chunked"
        assert row["context_window_tokens"] == 1024
        assert row["chunk_count"] == first.chunk_count
        source_chunk_calls = [
            prompt for prompt in llm.chunk_calls if "--- BEGIN CHUNK ---" in prompt
        ]
        assert len(source_chunk_calls) == first.chunk_count
        assert all("Chunk notes" not in prompt for prompt in source_chunk_calls)
        assert len(llm.analysis_calls) == 1
        assert "structured notes were extracted from every chunk" in llm.analysis_calls[0]
        chunk_calls = len(llm.chunk_calls)
        analysis_calls = len(llm.analysis_calls)

        second = analyze_chunked_text(
            db=db,
            prompt_path=config.paths.prompt,
            llm_config=profile,
            llm=llm,
            paper_id=paper_id,
            paper=paper_record,
            text=text,
            context_window_tokens=1024,
        )
        assert not second.outcome.created
        assert len(llm.chunk_calls) == chunk_calls
        assert len(llm.analysis_calls) == analysis_calls
        assert db._conn.execute("SELECT count(*) FROM analysis_chunks").fetchone()[0] == first.chunk_count


def test_chunk_reduction_and_validation_errors(analysis_result):
    llm = FakeChunkLLM(analysis_result)
    notes = [
        ChunkNotes(
            section=f"Section {index}",
            substantive_content="Substantive content " * 12,
            definitions=[],
            methods=[],
            results=[],
            assumptions=[],
            evidence=[],
            limitations=[],
            bibliography_titles=[],
        )
        for index in range(8)
    ]
    usage = {}
    encoded = _reduce_to_budget(llm, notes, 1400, usage)
    assert "Consolidated notes" in encoded
    assert usage["eval_count"] > 0

    invalid = FakeChunkLLM(analysis_result, invalid=True)
    with pytest.raises(LLMError, match="Invalid chunk summary"):
        _reduce_to_budget(invalid, notes, 1400, {})

    class OldClient:
        provider = "old"
        model = "old"

    from paperwatch.chunking import _generate_notes

    with pytest.raises(LLMError, match="does not support"):
        _generate_notes(OldClient(), "x")


def test_direct_prompt_fit_respects_context_and_source_limit(
    config_path, paper_record
):
    from paperwatch.chunking import direct_prompt_fit

    config = load_config(config_path)
    fits, prompt_chars, budget = direct_prompt_fit(
        prompt_path=config.paths.prompt,
        max_input_chars=5000,
        paper=paper_record,
        text="short source",
        scope="abstract",
        model_name="m",
        context_window_tokens=8192,
    )
    assert fits and 0 < prompt_chars < budget

    fits, prompt_chars, budget = direct_prompt_fit(
        prompt_path=config.paths.prompt,
        max_input_chars=5000,
        paper=paper_record,
        text="x" * 2000,
        scope="fulltext",
        model_name="m",
        context_window_tokens=512,
    )
    assert not fits and prompt_chars > budget

    fits, prompt_chars, _ = direct_prompt_fit(
        prompt_path=config.paths.prompt,
        max_input_chars=10,
        paper=paper_record,
        text="source is too long",
        scope="abstract",
        model_name="m",
        context_window_tokens=8192,
    )
    assert not fits and prompt_chars == 0
