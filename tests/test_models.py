from __future__ import annotations

import pytest
from pydantic import ValidationError

from paperwatch.models import (
    AnalysisResult,
    ChunkNotes,
    extract_arxiv_id,
    normalize_title,
    reconstruct_abstract,
    short_openalex_id,
    strip_doi,
)


def test_reconstruct_abstract_uses_positions():
    index = {"world": [1], "hello": [0], "again": [2]}
    assert reconstruct_abstract(index) == "hello world again"
    assert reconstruct_abstract(None) is None


def test_identifier_normalization():
    assert short_openalex_id("https://openalex.org/A123") == "A123"
    assert strip_doi("https://doi.org/10.1000/Test") == "10.1000/Test"
    assert extract_arxiv_id("https://arxiv.org/pdf/2607.12345v2.pdf") == "2607.12345"
    assert extract_arxiv_id("arXiv:cs/0701001v3") == "cs/0701001"


def test_title_normalization():
    assert normalize_title("  Minimal—Agents: Reliable ML! ") == "minimal agents reliable ml"


def test_analysis_result_enforces_exact_schema_and_invariants(analysis_result):
    payload = analysis_result.model_dump()

    invalid_kind = {**payload, "paper_kind": "physics"}
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(invalid_kind)

    invalid_setting = {**payload, "assumptions": ["Not a setting."]}
    with pytest.raises(ValidationError, match="Setting"):
        AnalysisResult.model_validate(invalid_setting)

    invalid_date = {**payload, "publication_date": "July 18, 2026"}
    with pytest.raises(ValidationError, match="YYYY-MM-DD"):
        AnalysisResult.model_validate(invalid_date)

    extra_key = {**payload, "title": payload["name"]}
    with pytest.raises(ValidationError, match="Extra inputs"):
        AnalysisResult.model_validate(extra_key)

    too_long = {**payload, "abstract_summary": "word " * 201}
    with pytest.raises(ValidationError, match="200 words"):
        AnalysisResult.model_validate(too_long)

    empty_author = {**payload, "authors": [""]}
    with pytest.raises(ValidationError, match="empty"):
        AnalysisResult.model_validate(empty_author)


def test_chunk_notes_allow_intermediate_detail_but_remain_bounded():
    detailed = "detail " * 400
    notes = ChunkNotes(
        section="Chunk 1",
        substantive_content=detailed,
        definitions=[detailed],
        methods=[],
        results=[],
        assumptions=[],
        evidence=[],
        limitations=[],
        bibliography_titles=[],
    )
    assert len(notes.substantive_content.split()) == 400

    with pytest.raises(ValidationError, match="1200 words"):
        ChunkNotes(
            section="Chunk 1",
            substantive_content="detail " * 1201,
            definitions=[],
            methods=[],
            results=[],
            assumptions=[],
            evidence=[],
            limitations=[],
            bibliography_titles=[],
        )


def test_text_and_identifier_edge_cases():
    from paperwatch.models import clean_text

    assert clean_text(None) is None
    assert clean_text("  \n ") is None
    assert short_openalex_id(None) is None
    assert short_openalex_id("https://openalex.org/a123/") == "A123"
    assert strip_doi("doi: 10.1000/ABC") == "10.1000/ABC"
    assert strip_doi(None) is None
    assert extract_arxiv_id(None, "not an arxiv link") is None


def test_unicode_title_normalization_and_identity_branches(paper_record):
    from dataclasses import replace

    assert normalize_title("Étude_日本語") == "étude 日本語"
    assert replace(paper_record, doi=None).identity_key.startswith("arxiv:")
    assert replace(paper_record, doi=None, arxiv_id=None).identity_key == "openalex:W456"
    title_only = replace(paper_record, doi=None, arxiv_id=None, openalex_id=None)
    assert title_only.identity_key.startswith("title:minimal agents")


def test_analysis_json_schema_has_exact_top_level_keys():
    from paperwatch.models import ANALYSIS_JSON_SCHEMA

    expected = {
        "name",
        "generated_by_model",
        "publication_date",
        "authors",
        "paper_kind",
        "source",
        "one_sentence_summary",
        "abstract_summary",
        "research_problem",
        "key_contributions",
        "assumptions",
        "definitions_and_notation",
        "methods_and_models",
        "main_results",
        "datasets",
        "supporting_evidence",
        "conclusion_summary",
        "limitations_and_open_questions",
        "extended_outline",
        "appendix_summaries",
        "notes",
    }
    assert set(ANALYSIS_JSON_SCHEMA["properties"]) == expected
    assert set(ANALYSIS_JSON_SCHEMA["required"]) == expected
    assert ANALYSIS_JSON_SCHEMA["additionalProperties"] is False
    assert "title" not in ANALYSIS_JSON_SCHEMA["properties"]
    for definition in ANALYSIS_JSON_SCHEMA["$defs"].values():
        assert definition["additionalProperties"] is False


def test_analysis_result_cardinality_and_length_limits(analysis_result):
    payload = analysis_result.model_dump()
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate({**payload, "key_contributions": ["Only one."]})
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate({**payload, "main_results": []})
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate({**payload, "one_sentence_summary": "x" * 501})


def test_model_attribution_uses_exposed_versions():
    from paperwatch.models import model_attribution

    assert model_attribution("qwen3:8b") == "qwen3:8b"
    assert model_attribution("gpt-4.1-2025-04-14") == "gpt-4.1-2025-04-14"
    assert model_attribution("gpt-test") == "gpt-test (version not exposed)"
