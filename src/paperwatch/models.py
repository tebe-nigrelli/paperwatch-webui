from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator


SPACE_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)
ARXIV_ID_PATTERN = r"(?:[a-z][a-z0-9.\-]*/\d{7}|\d{4}\.\d{4,5})"
ARXIV_RE = re.compile(
    rf"(?:"
    rf"(?:https?://)?(?:www\.|export\.)?arxiv\.org/(?:abs|pdf|html|format)/"
    rf"|arxiv\s*:\s*"
    rf"|10\.48550/arxiv\."
    rf"|arxiv\."
    rf")({ARXIV_ID_PATTERN})(?:v\d+)?(?:\.pdf)?",
    re.I,
)
BARE_ARXIV_RE = re.compile(rf"^({ARXIV_ID_PATTERN})(?:v\d+)?(?:\.pdf)?$", re.I)


def clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = SPACE_RE.sub(" ", value).strip()
    return cleaned or None


def normalize_title(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return NON_WORD_RE.sub(" ", normalized).strip()


def short_openalex_id(value: str | None) -> str | None:
    if not value:
        return None
    return value.rstrip("/").rsplit("/", 1)[-1].upper()


def strip_doi(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip().strip("<>[]{}")
    cleaned = re.sub(
        r"^https?://(?:(?:dx|www)\.)?doi\.org/", "", cleaned, flags=re.I
    )
    cleaned = re.sub(r"^doi\s*:?\s*", "", cleaned, flags=re.I)
    return cleaned.strip().rstrip(".,;") or None


def extract_arxiv_id(*values: str | None) -> str | None:
    for value in values:
        if not value:
            continue
        cleaned = value.strip().strip("<>[]{}")
        # An arXiv-issued DataCite DOI is just another representation of the
        # arXiv identifier. Normalizing the DOI first also handles doi: and
        # doi.org URL prefixes.
        doi = strip_doi(cleaned)
        candidates = [cleaned]
        if doi and doi != cleaned:
            candidates.append(doi)
        match = None
        for candidate in candidates:
            match = ARXIV_RE.search(candidate)
            if match is not None:
                break
        if match is None:
            match = BARE_ARXIV_RE.fullmatch(cleaned)
        if match:
            return re.sub(r"v\d+$", "", match.group(1), flags=re.I)
    return None


def reconstruct_abstract(index: dict[str, list[int]] | None) -> str | None:
    if not index:
        return None
    positioned: list[tuple[int, str]] = []
    for word, positions in index.items():
        positioned.extend((int(pos), word) for pos in positions)
    positioned.sort(key=lambda item: item[0])
    return clean_text(" ".join(word for _, word in positioned))


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def model_attribution(model: str) -> str:
    cleaned = model.strip()
    if re.search(r"\d{4}-\d{2}-\d{2}", cleaned) or ":" in cleaned or "@" in cleaned:
        return cleaned
    return f"{cleaned} (version not exposed)"


@dataclass(slots=True)
class AuthorCandidate:
    openalex_id: str
    name: str
    orcid: str | None = None
    institution: str | None = None
    works_count: int = 0
    cited_by_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PaperAuthor:
    openalex_id: str | None
    name: str
    position: str | None = None
    is_corresponding: bool = False


@dataclass(slots=True)
class PaperTopic:
    openalex_id: str
    name: str
    score: float | None = None
    is_primary: bool = False
    subfield_id: str | None = None
    subfield_name: str | None = None
    field_id: str | None = None
    field_name: str | None = None
    domain_id: str | None = None
    domain_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_openalex_topics(work: dict[str, Any]) -> list[PaperTopic]:
    """Return the ranked OpenAlex topics attached to a work."""

    primary = work.get("primary_topic")
    primary_id = (
        short_openalex_id(primary.get("id"))
        if isinstance(primary, dict)
        else None
    )
    raw_topics = work.get("topics")
    items = list(raw_topics) if isinstance(raw_topics, list) else []
    if isinstance(primary, dict):
        items.append(primary)

    topics: dict[str, PaperTopic] = {}
    order: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        topic_id = short_openalex_id(item.get("id"))
        name = clean_text(item.get("display_name"))
        if not topic_id or not topic_id.startswith("T") or not name:
            continue

        def hierarchy(level: str) -> tuple[str | None, str | None]:
            value = item.get(level)
            if not isinstance(value, dict):
                return None, None
            return (
                short_openalex_id(value.get("id")),
                clean_text(value.get("display_name")),
            )

        score_raw = item.get("score")
        try:
            score = float(score_raw) if score_raw is not None else None
        except (TypeError, ValueError):
            score = None
        subfield_id, subfield_name = hierarchy("subfield")
        field_id, field_name = hierarchy("field")
        domain_id, domain_name = hierarchy("domain")
        candidate = PaperTopic(
            openalex_id=topic_id,
            name=name,
            score=score,
            is_primary=topic_id == primary_id,
            subfield_id=subfield_id,
            subfield_name=subfield_name,
            field_id=field_id,
            field_name=field_name,
            domain_id=domain_id,
            domain_name=domain_name,
            metadata=item,
        )
        if topic_id not in topics:
            order.append(topic_id)
            topics[topic_id] = candidate
        else:
            existing = topics[topic_id]
            if candidate.is_primary or (existing.score is None and candidate.score is not None):
                candidate.is_primary = candidate.is_primary or existing.is_primary
                topics[topic_id] = candidate

    if primary_id and primary_id in topics:
        topics[primary_id].is_primary = True
    return [topics[topic_id] for topic_id in order]


@dataclass(slots=True)
class PaperRecord:
    openalex_id: str | None
    doi: str | None
    arxiv_id: str | None
    title: str
    abstract: str | None
    publication_date: str | None
    updated_date: str | None
    venue: str | None
    publication_url: str | None
    pdf_url: str | None
    abstract_source: str | None
    pdf_source: str | None
    authors: list[PaperAuthor]
    topics: list[PaperTopic] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_title(self) -> str:
        return normalize_title(self.title)

    @property
    def identity_key(self) -> str:
        if self.doi:
            return f"doi:{self.doi.lower()}"
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id.lower()}"
        if self.openalex_id:
            return f"openalex:{self.openalex_id.upper()}"
        year = (self.publication_date or "")[:4]
        return f"title:{self.normalized_title}:{year}"


def _word_limit(value: str, *, maximum: int, label: str) -> str:
    if len(value.split()) > maximum:
        raise ValueError(f"{label} must contain at most {maximum} words")
    return value


def _narrative_limit(value: str) -> str:
    return _word_limit(value, maximum=200, label="narrative strings")


def _chunk_narrative_limit(value: str) -> str:
    # Chunk notes are intermediate evidence, not final user-facing narrative fields.
    # They need enough room to preserve details before the reduction and final merge.
    return _word_limit(value, maximum=1200, label="chunk-note strings")


Narrative = Annotated[str, AfterValidator(_narrative_limit)]
ChunkNarrative = Annotated[str, AfterValidator(_chunk_narrative_limit)]
CollectionNarrative = Annotated[str, Field(max_length=100_000)]
OneSentence = Annotated[Narrative, Field(max_length=500)]
PaperKind = Literal[
    "mathematics",
    "machine_learning",
    "theoretical_machine_learning",
    "survey",
    "other",
]
MethodKind = Literal[
    "model", "algorithm", "proof_technique", "construction", "framework", "other"
]
ResultType = Literal[
    "theorem",
    "lemma",
    "proposition",
    "corollary",
    "empirical_finding",
    "algorithmic_result",
    "negative_result",
    "other",
]
EvidenceKind = Literal[
    "proof", "experiment", "benchmark", "ablation", "counterexample", "case_study"
]
AppendixRelevance = Literal["essential", "useful", "optional"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MethodAndModel(StrictModel):
    name: str = Field(min_length=1)
    kind: MethodKind
    purpose: Narrative
    description: Narrative
    primary_source_title: str | None


class MainResult(StrictModel):
    result_type: ResultType
    statement: Narrative
    conditions: Narrative | None
    significance: Narrative | None


class Dataset(StrictModel):
    name: str = Field(min_length=1)
    role: Narrative | None
    scope: Narrative | None
    preprocessing: Narrative | None


class SupportingEvidence(StrictModel):
    kind: EvidenceKind
    supports: Narrative
    approach: Narrative
    outcome: Narrative | None
    caveat: Narrative | None


class OutlineSection(StrictModel):
    section: str = Field(min_length=1)
    summary: Narrative


class AppendixSummary(StrictModel):
    appendix: str = Field(min_length=1)
    summary: Narrative
    relevance: AppendixRelevance


class AnalysisResult(StrictModel):
    name: str = Field(min_length=1)
    generated_by_model: str = Field(min_length=1)
    publication_date: str | None
    authors: list[str] = Field(min_length=1)
    paper_kind: PaperKind
    source: str | None
    one_sentence_summary: OneSentence
    abstract_summary: Narrative
    research_problem: Narrative
    key_contributions: list[Narrative] = Field(min_length=2, max_length=6)
    assumptions: list[Narrative] = Field(min_length=1)
    definitions_and_notation: Narrative | None
    methods_and_models: list[MethodAndModel]
    main_results: list[MainResult] = Field(min_length=1)
    datasets: list[Dataset]
    supporting_evidence: list[SupportingEvidence]
    conclusion_summary: Narrative
    limitations_and_open_questions: list[Narrative]
    extended_outline: list[OutlineSection]
    appendix_summaries: list[AppendixSummary]
    notes: Narrative | None

    @field_validator("publication_date")
    @classmethod
    def validate_publication_date(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("publication_date must use YYYY-MM-DD") from exc
        return value

    @field_validator("authors")
    @classmethod
    def validate_authors(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("authors cannot contain empty names")
        return values

    @model_validator(mode="after")
    def validate_setting_assumption(self) -> AnalysisResult:
        if not self.assumptions[0].startswith("Setting: "):
            raise ValueError('the first assumption must begin with "Setting: "')
        return self


ANALYSIS_JSON_SCHEMA: dict[str, Any] = AnalysisResult.model_json_schema()

class ChunkNotes(StrictModel):
    section: str = Field(min_length=1)
    substantive_content: ChunkNarrative
    definitions: list[ChunkNarrative]
    methods: list[ChunkNarrative]
    results: list[ChunkNarrative]
    assumptions: list[ChunkNarrative]
    evidence: list[ChunkNarrative]
    limitations: list[ChunkNarrative]
    bibliography_titles: list[str]


class BatchQueryItem(StrictModel):
    paper_id: int | None
    blog_entry_id: int | None = None
    content_type: Literal["paper", "blog"] = "paper"
    title: str = Field(min_length=1)
    output: CollectionNarrative


class BatchQueryResult(StrictModel):
    title: str = Field(min_length=1)
    summary: CollectionNarrative
    items: list[BatchQueryItem]


CHUNK_NOTES_JSON_SCHEMA: dict[str, Any] = ChunkNotes.model_json_schema()
BATCH_QUERY_JSON_SCHEMA: dict[str, Any] = BatchQueryResult.model_json_schema()


@dataclass(slots=True)
class LLMResponse:
    result: AnalysisResult
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
