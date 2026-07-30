from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperwatch.config import load_config
from paperwatch.models import (
    AnalysisResult,
    AuthorCandidate,
    PaperAuthor,
    PaperRecord,
    PaperTopic,
)


@pytest.fixture
def fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def openalex_author_json(fixture_dir: Path) -> dict:
    return json.loads((fixture_dir / "openalex_author.json").read_text())


@pytest.fixture
def openalex_work_json(fixture_dir: Path) -> dict:
    return json.loads((fixture_dir / "openalex_work.json").read_text())


@pytest.fixture
def author_candidate() -> AuthorCandidate:
    return AuthorCandidate(
        openalex_id="A123",
        name="Ada Lovelace",
        orcid="https://orcid.org/0000-0000-0000-0001",
        institution="Analytical Engine Institute",
        works_count=42,
        cited_by_count=9001,
    )


@pytest.fixture
def paper_record() -> PaperRecord:
    return PaperRecord(
        openalex_id="W456",
        doi="10.1000/example",
        arxiv_id="2607.12345",
        title="Minimal Agents for Reliable Machine Learning",
        abstract="We introduce a compact agent architecture.",
        publication_date="2026-07-18",
        updated_date="2026-07-20",
        venue="Journal of Minimal ML",
        publication_url="https://doi.org/10.1000/example",
        pdf_url="https://arxiv.org/pdf/2607.12345.pdf",
        abstract_source="openalex",
        pdf_source="openalex_best_oa",
        authors=[
            PaperAuthor("A123", "Ada Lovelace", "first", True),
            PaperAuthor("A999", "Alan Turing", "last", False),
        ],
        topics=[
            PaperTopic(
                openalex_id="T100",
                name="Machine Learning",
                score=0.98,
                is_primary=True,
                subfield_id="1702",
                subfield_name="Artificial Intelligence",
                field_id="17",
                field_name="Computer Science",
                domain_id="3",
                domain_name="Physical Sciences",
                metadata={"id": "https://openalex.org/T100"},
            ),
            PaperTopic(
                openalex_id="T200",
                name="Autonomous Agents",
                score=0.77,
                metadata={"id": "https://openalex.org/T200"},
            ),
        ],
        metadata={
            "id": "https://openalex.org/W456",
            "topics": [
                {
                    "id": "https://openalex.org/T100",
                    "display_name": "Machine Learning",
                    "score": 0.98,
                },
                {
                    "id": "https://openalex.org/T200",
                    "display_name": "Autonomous Agents",
                    "score": 0.77,
                },
            ],
            "primary_topic": {
                "id": "https://openalex.org/T100",
                "display_name": "Machine Learning",
                "score": 0.98,
            },
        },
    )


@pytest.fixture
def analysis_result() -> AnalysisResult:
    return AnalysisResult(
        name="Minimal Agents for Reliable Machine Learning",
        generated_by_model="test-model",
        publication_date="2026-07-18",
        authors=["Ada Lovelace", "Alan Turing"],
        paper_kind="machine_learning",
        source="10.1000/example",
        one_sentence_summary=(
            "The paper introduces a compact agent architecture and reports improved reliability "
            "on three benchmarks under a fixed evaluation protocol."
        ),
        abstract_summary=(
            "The work studies reliable machine-learning agents, proposes a compact architecture, "
            "and evaluates it on three benchmarks; the limited benchmark set constrains generality."
        ),
        research_problem=(
            "Given benchmark tasks and fixed observations, construct an agent that returns task "
            "predictions reliably while using a compact architecture."
        ),
        key_contributions=[
            "Introduces a compact agent architecture for reliability-focused evaluation.",
            "Measures the architecture against baselines on three benchmarks.",
        ],
        assumptions=[
            "Setting: Agents receive benchmark inputs, produce task predictions, and are evaluated "
            "under the paper's fixed data and metric protocol.",
            "The selected benchmarks are treated as representative of the target setting.",
        ],
        definitions_and_notation=None,
        methods_and_models=[
            {
                "name": "Compact agent architecture",
                "kind": "model",
                "purpose": "Produce reliable task predictions with a small architecture.",
                "description": "The model transforms benchmark inputs through the proposed compact agent pipeline and emits task predictions.",
                "primary_source_title": None,
            }
        ],
        main_results=[
            {
                "result_type": "empirical_finding",
                "statement": "The proposed architecture improves the reported reliability metric on the three evaluated benchmarks.",
                "conditions": "The paper's fixed datasets, baselines, and evaluation protocol.",
                "significance": "The result supports the method within the tested benchmark scope.",
            }
        ],
        datasets=[
            {
                "name": "Benchmark A",
                "role": "Evaluation benchmark.",
                "scope": "One of three reported tasks.",
                "preprocessing": None,
            }
        ],
        supporting_evidence=[
            {
                "kind": "benchmark",
                "supports": "The compact architecture improves reliability in the evaluated setting.",
                "approach": "The method is compared with reported baselines on three benchmarks.",
                "outcome": "The proposed method obtains the strongest reported reliability result.",
                "caveat": "Only three benchmarks are evaluated.",
            }
        ],
        conclusion_summary=(
            "The compact architecture improves reported reliability on the tested benchmarks, but "
            "the narrow evaluation prevents broad claims about other tasks or distributions."
        ),
        limitations_and_open_questions=["Evidence gap: Evaluation covers only three benchmarks."],
        extended_outline=[
            {
                "section": "1 Introduction and method",
                "summary": "The paper motivates reliable agents and defines the compact architecture used in evaluation.",
            },
            {
                "section": "2 Experiments",
                "summary": "The method is compared with baselines on three benchmarks using the reported reliability metric.",
            },
        ],
        appendix_summaries=[],
        notes=None,
    )


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    prompt = tmp_path / "prompts" / "analyze.txt"
    prompt.parent.mkdir()
    prompt.write_text(
        "Title: {title}\nAuthors: {authors}\nDate: {publication_date}\nVenue: {venue}\nSource: {source}\nModel: {model_name}\nScope: {scope}\n{text}",
        encoding="utf-8",
    )
    config = tmp_path / "config.toml"
    config.write_text(
        f"""
[paths]
database = "{(tmp_path / 'paperwatch.db').as_posix()}"
pdf_dir = "{(tmp_path / 'pdfs').as_posix()}"
prompt = "{prompt.as_posix()}"

[openalex]
api_key = "test-key"
base_url = "https://api.openalex.test"
timeout_seconds = 1

[arxiv]
enabled = true
base_url = "https://arxiv.test/api/query"
timeout_seconds = 1
min_interval_seconds = 0
title_match_threshold = 0.9

[fetch]
overlap_days = 7
initial_lookback_days = 30
max_papers_per_author = 100
download_pdfs = true
max_pdf_bytes = 1000000
max_pdf_chars = 10000

[llm]
provider = "ollama"
model = "test-model"
base_url = "http://ollama.test"
api_key = ""
timeout_seconds = 1
max_input_chars = 5000
analyze_pdfs = true
prompt_version = "test-v1"
""",
        encoding="utf-8",
    )
    return config


@pytest.fixture
def app_config(config_path: Path):
    return load_config(config_path)
