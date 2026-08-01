from __future__ import annotations

import json
import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from .models import (
    AuthorCandidate,
    PaperAuthor,
    PaperRecord,
    PaperTopic,
    json_dumps,
    parse_openalex_topics,
    short_openalex_id,
)


SCHEMA_VERSION = 13

DEFAULT_SUMMARY_QUERY_NAME = "Summarize"
DEFAULT_SUMMARY_QUERY_PROMPT = (
    "Summarize each selected paper. Cover the research problem, methods, main results, "
    "supporting evidence, limitations, and open questions. Keep every claim grounded in "
    "the selected source text and preserve the paper title and paper_id."
)


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


COMMON_SCHEMA = """
CREATE TABLE IF NOT EXISTS authors (
    id INTEGER PRIMARY KEY,
    openalex_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    orcid TEXT,
    institution TEXT,
    works_count INTEGER NOT NULL DEFAULT 0,
    cited_by_count INTEGER NOT NULL DEFAULT 0,
    enabled INTEGER NOT NULL DEFAULT 1,
    tracked INTEGER NOT NULL DEFAULT 1,
    added_at TEXT NOT NULL,
    last_checked_at TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY,
    identity_key TEXT UNIQUE NOT NULL,
    openalex_id TEXT UNIQUE,
    doi TEXT,
    arxiv_id TEXT,
    title TEXT NOT NULL,
    normalized_title TEXT NOT NULL,
    abstract TEXT,
    publication_date TEXT,
    updated_date TEXT,
    venue TEXT,
    publication_url TEXT,
    pdf_url TEXT,
    local_pdf_path TEXT,
    auto_analyze INTEGER NOT NULL DEFAULT 1,
    abstract_source TEXT,
    pdf_source TEXT,
    discovered_at TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_papers_arxiv ON papers(arxiv_id COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_papers_publication_date ON papers(publication_date);
CREATE INDEX IF NOT EXISTS idx_papers_title_year ON papers(normalized_title, publication_date);

CREATE TABLE IF NOT EXISTS paper_authors (
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    tracked_author_id INTEGER REFERENCES authors(id) ON DELETE SET NULL,
    openalex_author_id TEXT,
    name TEXT NOT NULL,
    author_order INTEGER NOT NULL,
    author_position TEXT,
    is_corresponding INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (paper_id, author_order)
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    authors_checked INTEGER NOT NULL DEFAULT 0,
    papers_found INTEGER NOT NULL DEFAULT 0,
    papers_analyzed INTEGER NOT NULL DEFAULT 0,
    pdfs_downloaded INTEGER NOT NULL DEFAULT 0,
    errors_count INTEGER NOT NULL DEFAULT 0,
    error_text TEXT
);

CREATE TABLE IF NOT EXISTS run_errors (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    author_openalex_id TEXT,
    paper_identity_key TEXT,
    stage TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

BLOGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS blogs (
    id INTEGER PRIMARY KEY,
    url TEXT NOT NULL UNIQUE,
    title TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    etag TEXT,
    last_modified TEXT,
    feed_url TEXT,
    feed_etag TEXT,
    feed_last_modified TEXT,
    feed_initialized INTEGER NOT NULL DEFAULT 0,
    feed_discovery_checked INTEGER NOT NULL DEFAULT 0,
    last_fetch_method TEXT CHECK (last_fetch_method IN ('rss', 'snapshot')),
    last_checked_at TEXT,
    last_status INTEGER,
    last_error TEXT,
    added_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS author_blogs (
    author_id INTEGER NOT NULL REFERENCES authors(id) ON DELETE CASCADE,
    blog_id INTEGER NOT NULL REFERENCES blogs(id) ON DELETE CASCADE,
    label TEXT,
    relationship TEXT,
    PRIMARY KEY (author_id, blog_id)
);
CREATE INDEX IF NOT EXISTS idx_author_blogs_blog ON author_blogs(blog_id, author_id);
CREATE TABLE IF NOT EXISTS blog_entries (
    id INTEGER PRIMARY KEY,
    blog_id INTEGER NOT NULL REFERENCES blogs(id) ON DELETE CASCADE,
    previous_entry_id INTEGER REFERENCES blog_entries(id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    diff_text TEXT NOT NULL DEFAULT '',
    change_kind TEXT NOT NULL CHECK (change_kind IN ('baseline', 'changed', 'reset')),
    entry_kind TEXT NOT NULL DEFAULT 'snapshot' CHECK (entry_kind IN ('rss', 'snapshot')),
    external_id TEXT,
    source_url TEXT,
    published_at TEXT,
    fetched_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_blog_entries_feed ON blog_entries(fetched_at DESC, id DESC);
"""


ANALYSES_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    scope TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    input_chars INTEGER NOT NULL,
    analysis_strategy TEXT NOT NULL DEFAULT 'direct',
    context_window_tokens INTEGER,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    name TEXT NOT NULL,
    generated_by_model TEXT NOT NULL,
    publication_date TEXT,
    authors TEXT NOT NULL,
    paper_kind TEXT NOT NULL CHECK (
        paper_kind IN ('mathematics', 'machine_learning', 'theoretical_machine_learning', 'survey', 'other')
    ),
    source TEXT,
    one_sentence_summary TEXT NOT NULL,
    abstract_summary TEXT NOT NULL,
    research_problem TEXT NOT NULL,
    key_contributions TEXT NOT NULL,
    assumptions TEXT NOT NULL,
    definitions_and_notation TEXT,
    methods_and_models TEXT NOT NULL,
    main_results TEXT NOT NULL,
    datasets TEXT NOT NULL,
    supporting_evidence TEXT NOT NULL,
    conclusion_summary TEXT NOT NULL,
    limitations_and_open_questions TEXT NOT NULL,
    extended_outline TEXT NOT NULL,
    appendix_summaries TEXT NOT NULL,
    notes TEXT,
    result_json TEXT NOT NULL,
    usage_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE (paper_id, scope, provider, model, prompt_version, content_sha256)
);
CREATE INDEX IF NOT EXISTS idx_analyses_paper_scope ON analyses(paper_id, scope);
CREATE INDEX IF NOT EXISTS idx_analyses_kind ON analyses(paper_kind);
"""

ANALYSIS_TASKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_tasks (
    id INTEGER PRIMARY KEY,
    batch_id TEXT NOT NULL,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    scope TEXT NOT NULL CHECK (scope IN ('abstract', 'fulltext')),
    profile TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'success', 'failed')),
    analysis_id INTEGER REFERENCES analyses(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    error_text TEXT,
    analysis_strategy TEXT NOT NULL DEFAULT 'direct',
    context_window_tokens INTEGER,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    UNIQUE (batch_id, profile)
);
CREATE INDEX IF NOT EXISTS idx_analysis_tasks_paper ON analysis_tasks(paper_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_tasks_batch ON analysis_tasks(batch_id, id);
"""

EXTRA_SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_chunks (
    id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    context_window_tokens INTEGER NOT NULL,
    summary_json TEXT NOT NULL,
    usage_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE (
        paper_id, provider, model, prompt_version, content_sha256,
        chunk_index, context_window_tokens
    )
);
CREATE INDEX IF NOT EXISTS idx_analysis_chunks_paper
    ON analysis_chunks(paper_id, provider, model);

CREATE TABLE IF NOT EXISTS author_tags (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL COLLATE NOCASE UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS author_tag_links (
    author_id INTEGER NOT NULL REFERENCES authors(id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES author_tags(id) ON DELETE CASCADE,
    PRIMARY KEY (author_id, tag_id)
);
CREATE INDEX IF NOT EXISTS idx_author_tag_links_tag ON author_tag_links(tag_id, author_id);

CREATE TABLE IF NOT EXISTS saved_queries (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('papers', 'summaries')),
    search_query TEXT NOT NULL DEFAULT '',
    prompt TEXT NOT NULL,
    is_builtin INTEGER NOT NULL DEFAULT 0,
    auto_run_papers INTEGER NOT NULL DEFAULT 0,
    auto_run_blogs INTEGER NOT NULL DEFAULT 0,
    model_profiles_json TEXT NOT NULL DEFAULT '[]',
    context_window_tokens INTEGER NOT NULL DEFAULT 16384,
    paper_source_scope TEXT NOT NULL DEFAULT 'auto'
        CHECK (paper_source_scope IN ('auto', 'abstract', 'fulltext')),
    pdf_strategy TEXT NOT NULL DEFAULT 'chunked'
        CHECK (pdf_strategy IN ('auto', 'direct', 'chunked')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_run_at TEXT
);
CREATE TABLE IF NOT EXISTS query_runs (
    id INTEGER PRIMARY KEY,
    batch_id TEXT NOT NULL,
    query_id INTEGER NOT NULL REFERENCES saved_queries(id) ON DELETE CASCADE,
    profile TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'success', 'failed')),
    context_window_tokens INTEGER NOT NULL,
    selection_json TEXT NOT NULL DEFAULT '{}',
    matched_count INTEGER NOT NULL DEFAULT 0,
    input_chars INTEGER NOT NULL DEFAULT 0,
    output_json TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    error_text TEXT,
    UNIQUE (batch_id, profile)
);
CREATE INDEX IF NOT EXISTS idx_query_runs_query ON query_runs(query_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_query_runs_batch ON query_runs(batch_id, id);
"""

TOPICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS topics (
    openalex_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    subfield_id TEXT,
    subfield_name TEXT,
    field_id TEXT,
    field_name TEXT,
    domain_id TEXT,
    domain_name TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_topics_display_name
    ON topics(display_name COLLATE NOCASE);

CREATE TABLE IF NOT EXISTS paper_topics (
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    topic_id TEXT NOT NULL REFERENCES topics(openalex_id) ON DELETE CASCADE,
    score REAL,
    is_primary INTEGER NOT NULL DEFAULT 0,
    topic_order INTEGER NOT NULL,
    PRIMARY KEY (paper_id, topic_id)
);
CREATE INDEX IF NOT EXISTS idx_paper_topics_topic ON paper_topics(topic_id, paper_id);
CREATE INDEX IF NOT EXISTS idx_paper_topics_paper ON paper_topics(paper_id, topic_order);
"""

SCHEMA = (
    COMMON_SCHEMA + ANALYSES_SCHEMA + ANALYSIS_TASKS_SCHEMA + EXTRA_SCHEMA
    + TOPICS_SCHEMA + BLOGS_SCHEMA
)


class DatabaseError(RuntimeError):
    pass


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=10)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA busy_timeout = 10000")
        version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        if version > SCHEMA_VERSION:
            self._conn.close()
            raise DatabaseError(
                f"Database schema version {version} is newer than supported version {SCHEMA_VERSION}"
            )

        self._conn.executescript(COMMON_SCHEMA)
        self._ensure_author_tracking_column()
        self._ensure_paper_columns()
        if version == 1:
            self._migrate_v1_analyses()
        elif version == 0 and self._is_legacy_analyses_table():
            self._migrate_v1_analyses()
        else:
            self._conn.executescript(ANALYSES_SCHEMA)
        self._conn.executescript(ANALYSIS_TASKS_SCHEMA)
        self._ensure_analysis_columns()
        self._conn.executescript(EXTRA_SCHEMA)
        self._ensure_query_columns()
        self._ensure_default_saved_queries()
        self._conn.executescript(TOPICS_SCHEMA)
        self._conn.executescript(BLOGS_SCHEMA)
        self._ensure_blog_columns()
        self._ensure_run_blog_columns()
        if version < 6:
            self._backfill_paper_topics()
        self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._conn.commit()

    def _ensure_run_blog_columns(self) -> None:
        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(runs)").fetchall()
        }
        for name in ("blogs_checked", "blog_updates_found"):
            if name not in columns:
                self._conn.execute(
                    f"ALTER TABLE runs ADD COLUMN {name} INTEGER NOT NULL DEFAULT 0"
                )

    def _ensure_blog_columns(self) -> None:
        blog_columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(blogs)").fetchall()
        }
        blog_definitions = {
            "feed_url": "TEXT",
            "feed_etag": "TEXT",
            "feed_last_modified": "TEXT",
            "feed_initialized": "INTEGER NOT NULL DEFAULT 0",
            "feed_discovery_checked": "INTEGER NOT NULL DEFAULT 0",
            "last_fetch_method": "TEXT",
        }
        for name, definition in blog_definitions.items():
            if name not in blog_columns:
                self._conn.execute(f"ALTER TABLE blogs ADD COLUMN {name} {definition}")

        entry_columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(blog_entries)").fetchall()
        }
        entry_definitions = {
            "entry_kind": "TEXT NOT NULL DEFAULT 'snapshot'",
            "external_id": "TEXT",
            "source_url": "TEXT",
            "published_at": "TEXT",
        }
        for name, definition in entry_definitions.items():
            if name not in entry_columns:
                self._conn.execute(
                    f"ALTER TABLE blog_entries ADD COLUMN {name} {definition}"
                )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_blog_entries_external
            ON blog_entries(blog_id, entry_kind, external_id, id DESC)
            """
        )

    def _ensure_author_tracking_column(self) -> None:
        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(authors)").fetchall()
        }
        if "tracked" not in columns:
            self._conn.execute(
                "ALTER TABLE authors ADD COLUMN tracked INTEGER NOT NULL DEFAULT 1"
            )

    def _ensure_paper_columns(self) -> None:
        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(papers)").fetchall()
        }
        if "auto_analyze" not in columns:
            self._conn.execute(
                "ALTER TABLE papers ADD COLUMN auto_analyze INTEGER NOT NULL DEFAULT 1"
            )

    def _ensure_analysis_columns(self) -> None:
        additions = {
            "analyses": {
                "analysis_strategy": "TEXT NOT NULL DEFAULT 'direct'",
                "context_window_tokens": "INTEGER",
                "chunk_count": "INTEGER NOT NULL DEFAULT 0",
            },
            "analysis_tasks": {
                "analysis_strategy": "TEXT NOT NULL DEFAULT 'direct'",
                "context_window_tokens": "INTEGER",
                "chunk_count": "INTEGER NOT NULL DEFAULT 0",
            },
        }
        for table, columns in additions.items():
            existing = {
                row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for name, definition in columns.items():
                if name not in existing:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    def _ensure_query_columns(self) -> None:
        run_columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(query_runs)").fetchall()
        }
        if "selection_json" not in run_columns:
            self._conn.execute(
                "ALTER TABLE query_runs ADD COLUMN selection_json TEXT NOT NULL DEFAULT '{}'"
            )
        query_columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(saved_queries)").fetchall()
        }
        if "is_builtin" not in query_columns:
            self._conn.execute(
                "ALTER TABLE saved_queries ADD COLUMN is_builtin INTEGER NOT NULL DEFAULT 0"
            )
        query_additions = {
            "auto_run_papers": "INTEGER NOT NULL DEFAULT 0",
            "auto_run_blogs": "INTEGER NOT NULL DEFAULT 0",
            "model_profiles_json": "TEXT NOT NULL DEFAULT '[]'",
            "context_window_tokens": "INTEGER NOT NULL DEFAULT 16384",
            "paper_source_scope": "TEXT NOT NULL DEFAULT 'auto'",
            "pdf_strategy": "TEXT NOT NULL DEFAULT 'chunked'",
        }
        for name, definition in query_additions.items():
            if name not in query_columns:
                self._conn.execute(
                    f"ALTER TABLE saved_queries ADD COLUMN {name} {definition}"
                )

    def _ensure_default_saved_queries(self) -> None:
        row = self._conn.execute(
            "SELECT id FROM saved_queries WHERE is_builtin=1 ORDER BY id LIMIT 1"
        ).fetchone()
        if row is not None:
            return
        existing = self._conn.execute(
            "SELECT id FROM saved_queries WHERE lower(name)=lower(?) ORDER BY id LIMIT 1",
            (DEFAULT_SUMMARY_QUERY_NAME,),
        ).fetchone()
        if existing is not None:
            self._conn.execute(
                "UPDATE saved_queries SET name=?, is_builtin=1 WHERE id=?",
                (DEFAULT_SUMMARY_QUERY_NAME, int(existing["id"])),
            )
            return
        now = utcnow()
        self._conn.execute(
            """
            INSERT INTO saved_queries(
                name, source_type, search_query, prompt, is_builtin, created_at, updated_at
            ) VALUES (?, 'papers', '', ?, 1, ?, ?)
            """,
            (DEFAULT_SUMMARY_QUERY_NAME, DEFAULT_SUMMARY_QUERY_PROMPT, now, now),
        )

    def _backfill_paper_topics(self) -> None:
        rows = self._conn.execute(
            "SELECT id, metadata_json FROM papers ORDER BY id"
        ).fetchall()
        for row in rows:
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(metadata, dict):
                continue
            topics = parse_openalex_topics(metadata)
            if topics:
                self._sync_paper_topics(self._conn, int(row["id"]), topics)

    @staticmethod
    def _sync_paper_topics(
        conn: sqlite3.Connection, paper_id: int, topics: list[PaperTopic]
    ) -> None:
        conn.execute("DELETE FROM paper_topics WHERE paper_id = ?", (paper_id,))
        now = utcnow()
        for topic_order, topic in enumerate(topics):
            conn.execute(
                """
                INSERT INTO topics (
                    openalex_id, display_name, subfield_id, subfield_name,
                    field_id, field_name, domain_id, domain_name,
                    metadata_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(openalex_id) DO UPDATE SET
                    display_name=excluded.display_name,
                    subfield_id=COALESCE(excluded.subfield_id, topics.subfield_id),
                    subfield_name=COALESCE(excluded.subfield_name, topics.subfield_name),
                    field_id=COALESCE(excluded.field_id, topics.field_id),
                    field_name=COALESCE(excluded.field_name, topics.field_name),
                    domain_id=COALESCE(excluded.domain_id, topics.domain_id),
                    domain_name=COALESCE(excluded.domain_name, topics.domain_name),
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    topic.openalex_id,
                    topic.name,
                    topic.subfield_id,
                    topic.subfield_name,
                    topic.field_id,
                    topic.field_name,
                    topic.domain_id,
                    topic.domain_name,
                    json_dumps(topic.metadata),
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO paper_topics (
                    paper_id, topic_id, score, is_primary, topic_order
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    paper_id,
                    topic.openalex_id,
                    topic.score,
                    1 if topic.is_primary else 0,
                    topic_order,
                ),
            )

    def _is_legacy_analyses_table(self) -> bool:
        exists = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='analyses'"
        ).fetchone()
        if not exists:
            return False
        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(analyses)").fetchall()
        }
        return "relevance_score" in columns and "one_sentence_summary" not in columns

    def _migrate_v1_analyses(self) -> None:
        archive_exists = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='analyses_v1_archive'"
        ).fetchone()
        if self._is_legacy_analyses_table():
            if archive_exists:
                raise DatabaseError("Cannot migrate analyses: analyses_v1_archive already exists")
            self._conn.execute("ALTER TABLE analyses RENAME TO analyses_v1_archive")
        self._conn.executescript(ANALYSES_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    @staticmethod
    def _author_id(value: str) -> str:
        identifier = short_openalex_id(value)
        if not identifier:
            raise ValueError("Invalid OpenAlex author id")
        return identifier

    @staticmethod
    def _topic_id(value: str) -> str:
        identifier = short_openalex_id(value)
        if not identifier or not identifier.startswith("T"):
            raise ValueError("Invalid OpenAlex topic id")
        return identifier

    def add_author(self, author: AuthorCandidate) -> int:
        identifier = self._author_id(author.openalex_id)
        now = utcnow()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO authors (
                    openalex_id, name, orcid, institution, works_count,
                    cited_by_count, enabled, tracked, added_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, 1, 1, ?, ?)
                ON CONFLICT(openalex_id) DO UPDATE SET
                    name=excluded.name,
                    orcid=excluded.orcid,
                    institution=excluded.institution,
                    works_count=excluded.works_count,
                    cited_by_count=excluded.cited_by_count,
                    enabled=1,
                    tracked=1,
                    metadata_json=excluded.metadata_json
                """,
                (
                    identifier,
                    author.name,
                    author.orcid,
                    author.institution,
                    author.works_count,
                    author.cited_by_count,
                    now,
                    json_dumps(author.metadata),
                ),
            )
            row = conn.execute("SELECT id FROM authors WHERE openalex_id = ?", (identifier,)).fetchone()
            assert row is not None
            conn.execute(
                """
                UPDATE paper_authors
                SET tracked_author_id = ?
                WHERE openalex_author_id = ?
                """,
                (int(row["id"]), identifier),
            )
        return int(row["id"])

    def add_manual_author(self, name: str, institution: str | None = None) -> int:
        cleaned_name = " ".join(name.strip().split())
        if not cleaned_name:
            raise ValueError("Author name is required")
        if len(cleaned_name) > 300:
            raise ValueError("Author name is too long")
        cleaned_institution = " ".join((institution or "").strip().split()) or None
        digest = hashlib.sha256(cleaned_name.casefold().encode("utf-8")).hexdigest()[:16]
        identifier = f"LOCAL-{digest}".upper()
        now = utcnow()
        metadata = {"manual": True, "papers_enabled": False}
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO authors (
                    openalex_id, name, institution, enabled, tracked, added_at, metadata_json
                ) VALUES (?, ?, ?, 0, 1, ?, ?)
                ON CONFLICT(openalex_id) DO UPDATE SET
                    name=excluded.name,
                    institution=excluded.institution,
                    tracked=1,
                    metadata_json=excluded.metadata_json
                """,
                (identifier, cleaned_name, cleaned_institution, now, json_dumps(metadata)),
            )
            row = conn.execute(
                "SELECT id FROM authors WHERE openalex_id=?", (identifier,)
            ).fetchone()
        assert row is not None
        return int(row["id"])

    def find_authors_by_name(self, name: str) -> list[sqlite3.Row]:
        cleaned = " ".join(name.strip().split())
        if not cleaned:
            return []
        return list(
            self._conn.execute(
                "SELECT * FROM authors WHERE tracked=1 AND name=? COLLATE NOCASE ORDER BY id",
                (cleaned,),
            ).fetchall()
        )

    def list_authors(
        self,
        enabled_only: bool = False,
        tracked_only: bool = False,
    ) -> list[sqlite3.Row]:
        clauses: list[str] = []
        if enabled_only:
            clauses.extend(("enabled = 1", "tracked = 1"))
        elif tracked_only:
            clauses.append("tracked = 1")
        sql = "SELECT * FROM authors"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY name COLLATE NOCASE"
        return list(self._conn.execute(sql).fetchall())

    def get_author(self, openalex_id: str) -> sqlite3.Row | None:
        identifier = self._author_id(openalex_id)
        return self._conn.execute(
            "SELECT * FROM authors WHERE openalex_id = ?", (identifier,)
        ).fetchone()

    def set_author_enabled(self, openalex_id: str, enabled: bool) -> bool:
        identifier = self._author_id(openalex_id)
        with self.transaction() as conn:
            cur = conn.execute(
                "UPDATE authors SET enabled = ?, tracked = 1 WHERE openalex_id = ?",
                (1 if enabled else 0, identifier),
            )
        return cur.rowcount > 0

    def remove_author(self, openalex_id: str) -> bool:
        """Archive an author without deleting identity or paper relationships."""
        identifier = self._author_id(openalex_id)
        with self.transaction() as conn:
            cur = conn.execute(
                "UPDATE authors SET enabled = 0, tracked = 0 WHERE openalex_id = ?",
                (identifier,),
            )
        return cur.rowcount > 0

    def mark_author_checked(self, openalex_id: str, checked_at: str | None = None) -> None:
        identifier = self._author_id(openalex_id)
        with self.transaction() as conn:
            conn.execute(
                "UPDATE authors SET last_checked_at = ? WHERE openalex_id = ?",
                (checked_at or utcnow(), identifier),
            )

    def add_author_blog(
        self,
        openalex_id: str,
        url: str,
        *,
        label: str = "",
        relationship: str = "",
    ) -> int:
        identifier = self._author_id(openalex_id)
        cleaned_url = url.strip()
        if not cleaned_url:
            raise ValueError("Blog URL is required")
        cleaned_label = " ".join(label.strip().split())[:300] or None
        cleaned_relationship = " ".join(relationship.strip().split())[:300] or None
        with self.transaction() as conn:
            author = conn.execute(
                "SELECT id FROM authors WHERE openalex_id=? AND tracked=1", (identifier,)
            ).fetchone()
            if author is None:
                raise ValueError("Author not found")
            conn.execute(
                """
                INSERT INTO blogs(url, title, enabled, added_at)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(url) DO UPDATE SET enabled=1
                """,
                (cleaned_url, cleaned_label, utcnow()),
            )
            blog = conn.execute("SELECT id FROM blogs WHERE url=?", (cleaned_url,)).fetchone()
            assert blog is not None
            conn.execute(
                """
                INSERT INTO author_blogs(author_id, blog_id, label, relationship)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(author_id, blog_id) DO UPDATE SET
                    label=excluded.label,
                    relationship=excluded.relationship
                """,
                (int(author["id"]), int(blog["id"]), cleaned_label, cleaned_relationship),
            )
        return int(blog["id"])

    def remove_author_blog(self, openalex_id: str, blog_id: int) -> bool:
        identifier = self._author_id(openalex_id)
        with self.transaction() as conn:
            author = conn.execute(
                "SELECT id FROM authors WHERE openalex_id=?", (identifier,)
            ).fetchone()
            if author is None:
                return False
            cur = conn.execute(
                "DELETE FROM author_blogs WHERE author_id=? AND blog_id=?",
                (int(author["id"]), blog_id),
            )
        return cur.rowcount > 0

    def author_blogs(self, openalex_id: str) -> list[sqlite3.Row]:
        identifier = self._author_id(openalex_id)
        return list(
            self._conn.execute(
                """
                SELECT b.*, ab.label, ab.relationship,
                    (SELECT count(*) FROM blog_entries be
                     WHERE be.blog_id=b.id AND be.change_kind != 'baseline') AS entry_count
                FROM authors a
                JOIN author_blogs ab ON ab.author_id=a.id
                JOIN blogs b ON b.id=ab.blog_id
                WHERE a.openalex_id=?
                ORDER BY COALESCE(ab.label, b.title, b.url) COLLATE NOCASE
                """,
                (identifier,),
            ).fetchall()
        )

    def list_enabled_blogs(self) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                """
                SELECT b.*
                FROM blogs b
                WHERE b.enabled=1 AND EXISTS (
                    SELECT 1 FROM author_blogs ab
                    JOIN authors a ON a.id=ab.author_id
                    WHERE ab.blog_id=b.id AND a.tracked=1
                )
                ORDER BY b.id
                """
            ).fetchall()
        )

    def latest_blog_entry(
        self,
        blog_id: int,
        *,
        entry_kind: str = "snapshot",
        external_id: str | None = None,
    ) -> sqlite3.Row | None:
        if external_id is None:
            return self._conn.execute(
                """
                SELECT * FROM blog_entries
                WHERE blog_id=? AND entry_kind=?
                ORDER BY id DESC LIMIT 1
                """,
                (blog_id, entry_kind),
            ).fetchone()
        return self._conn.execute(
            """
            SELECT * FROM blog_entries
            WHERE blog_id=? AND entry_kind=? AND external_id=?
            ORDER BY id DESC LIMIT 1
            """,
            (blog_id, entry_kind, external_id),
        ).fetchone()

    def save_blog_entry(
        self,
        *,
        blog_id: int,
        title: str,
        content_sha256: str,
        normalized_text: str,
        diff_text: str,
        change_kind: str,
        previous_entry_id: int | None,
        entry_kind: str = "snapshot",
        external_id: str | None = None,
        source_url: str | None = None,
        published_at: str | None = None,
    ) -> int:
        with self.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO blog_entries(
                    blog_id, previous_entry_id, title, content_sha256,
                    normalized_text, diff_text, change_kind, entry_kind,
                    external_id, source_url, published_at, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    blog_id,
                    previous_entry_id,
                    title,
                    content_sha256,
                    normalized_text,
                    diff_text,
                    change_kind,
                    entry_kind,
                    external_id,
                    source_url,
                    published_at,
                    utcnow(),
                ),
            )
        return int(cur.lastrowid)

    def mark_blog_checked(
        self,
        blog_id: int,
        *,
        status: int | None,
        etag: str | None = None,
        last_modified: str | None = None,
        title: str | None = None,
        error: str | None = None,
        fetch_method: str | None = None,
        feed_url: str | None = None,
        feed_initialized: bool | None = None,
        feed_discovery_checked: bool | None = None,
    ) -> None:
        with self.transaction() as conn:
            if fetch_method == "rss":
                conn.execute(
                    """
                    UPDATE blogs SET last_checked_at=?, last_status=?, last_error=?,
                        feed_etag=COALESCE(?, feed_etag),
                        feed_last_modified=COALESCE(?, feed_last_modified),
                        feed_url=COALESCE(?, feed_url), title=COALESCE(?, title),
                        last_fetch_method='rss',
                        feed_initialized=COALESCE(?, feed_initialized),
                        feed_discovery_checked=COALESCE(?, feed_discovery_checked)
                    WHERE id=?
                    """,
                    (
                        utcnow(),
                        status,
                        error,
                        etag,
                        last_modified,
                        feed_url,
                        title,
                        int(feed_initialized) if feed_initialized is not None else None,
                        (
                            int(feed_discovery_checked)
                            if feed_discovery_checked is not None
                            else None
                        ),
                        blog_id,
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE blogs SET last_checked_at=?, last_status=?, last_error=?,
                        etag=COALESCE(?, etag), last_modified=COALESCE(?, last_modified),
                        feed_url=COALESCE(?, feed_url), title=COALESCE(?, title),
                        last_fetch_method=COALESCE(?, last_fetch_method),
                        feed_initialized=COALESCE(?, feed_initialized),
                        feed_discovery_checked=COALESCE(?, feed_discovery_checked)
                    WHERE id=?
                    """,
                    (
                        utcnow(),
                        status,
                        error,
                        etag,
                        last_modified,
                        feed_url,
                        title,
                        fetch_method,
                        int(feed_initialized) if feed_initialized is not None else None,
                        (
                            int(feed_discovery_checked)
                            if feed_discovery_checked is not None
                            else None
                        ),
                        blog_id,
                    ),
                )

    def list_blog_entries(self, limit: int | None = 100) -> list[sqlite3.Row]:
        sql = """
            SELECT be.*, b.url, COALESCE(b.title, be.title) AS blog_title,
                (SELECT group_concat(name, ', ') FROM (
                    SELECT DISTINCT a.name AS name
                    FROM author_blogs ab JOIN authors a ON a.id=ab.author_id
                    WHERE ab.blog_id=b.id ORDER BY a.name COLLATE NOCASE
                )) AS author_names
            FROM blog_entries be JOIN blogs b ON b.id=be.blog_id
            WHERE be.change_kind != 'baseline'
            ORDER BY be.fetched_at DESC, be.id DESC
        """
        if limit is None:
            return list(self._conn.execute(sql).fetchall())
        return list(self._conn.execute(sql + " LIMIT ?", (limit,)).fetchall())

    def blog_entries_by_ids(self, entry_ids: Iterable[int]) -> list[sqlite3.Row]:
        ids: list[int] = []
        for value in entry_ids:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0 and parsed not in ids:
                ids.append(parsed)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"""
            SELECT be.*, b.url, COALESCE(b.title, be.title) AS blog_title,
                (SELECT group_concat(name, ', ') FROM (
                    SELECT DISTINCT a.name AS name
                    FROM author_blogs ab JOIN authors a ON a.id=ab.author_id
                    WHERE ab.blog_id=b.id ORDER BY a.name COLLATE NOCASE
                )) AS author_names
            FROM blog_entries be JOIN blogs b ON b.id=be.blog_id
            WHERE be.id IN ({placeholders}) AND be.change_kind != 'baseline'
            """,
            ids,
        ).fetchall()
        by_id = {int(row["id"]): row for row in rows}
        return [by_id[value] for value in ids if value in by_id]

    def get_blog_entry(self, entry_id: int) -> sqlite3.Row | None:
        return self._conn.execute(
            """
            SELECT be.*, b.url, COALESCE(b.title, be.title) AS blog_title,
                (SELECT group_concat(name, ', ') FROM (
                    SELECT DISTINCT a.name AS name
                    FROM author_blogs ab JOIN authors a ON a.id=ab.author_id
                    WHERE ab.blog_id=b.id ORDER BY a.name COLLATE NOCASE
                )) AS author_names
            FROM blog_entries be JOIN blogs b ON b.id=be.blog_id
            WHERE be.id=?
            """,
            (entry_id,),
        ).fetchone()

    def _find_existing_paper(self, paper: PaperRecord) -> sqlite3.Row | None:
        row = self._conn.execute(
            "SELECT * FROM papers WHERE identity_key = ?", (paper.identity_key,)
        ).fetchone()
        if row:
            return row
        if paper.openalex_id:
            row = self._conn.execute(
                "SELECT * FROM papers WHERE openalex_id = ?", (paper.openalex_id,)
            ).fetchone()
            if row:
                return row
        if paper.doi:
            row = self._conn.execute(
                "SELECT * FROM papers WHERE doi = ? COLLATE NOCASE", (paper.doi,)
            ).fetchone()
            if row:
                return row
        if paper.arxiv_id:
            row = self._conn.execute(
                "SELECT * FROM papers WHERE arxiv_id = ? COLLATE NOCASE", (paper.arxiv_id,)
            ).fetchone()
            if row:
                return row
        year = (paper.publication_date or "")[:4]
        if paper.normalized_title and year:
            return self._conn.execute(
                """
                SELECT * FROM papers
                WHERE normalized_title = ? AND substr(publication_date, 1, 4) = ?
                ORDER BY id LIMIT 1
                """,
                (paper.normalized_title, year),
            ).fetchone()
        return None

    def upsert_paper(
        self, paper: PaperRecord, *, auto_analyze: bool = True
    ) -> tuple[int, bool]:
        existing = self._find_existing_paper(paper)
        now = utcnow()
        metadata = json_dumps(paper.metadata)

        with self.transaction() as conn:
            if existing is None:
                cur = conn.execute(
                    """
                    INSERT INTO papers (
                        identity_key, openalex_id, doi, arxiv_id, title,
                        normalized_title, abstract, publication_date, updated_date,
                        venue, publication_url, pdf_url, auto_analyze, abstract_source, pdf_source,
                        discovered_at, fetched_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        paper.identity_key,
                        paper.openalex_id,
                        paper.doi,
                        paper.arxiv_id,
                        paper.title,
                        paper.normalized_title,
                        paper.abstract,
                        paper.publication_date,
                        paper.updated_date,
                        paper.venue,
                        paper.publication_url,
                        paper.pdf_url,
                        1 if auto_analyze else 0,
                        paper.abstract_source,
                        paper.pdf_source,
                        now,
                        now,
                        metadata,
                    ),
                )
                paper_id = int(cur.lastrowid)
                is_new = True
            else:
                paper_id = int(existing["id"])
                conn.execute(
                    """
                    UPDATE papers SET
                        openalex_id=COALESCE(?, openalex_id),
                        doi=COALESCE(?, doi), arxiv_id=COALESCE(?, arxiv_id),
                        title=?, normalized_title=?,
                        abstract=COALESCE(?, abstract),
                        publication_date=COALESCE(?, publication_date),
                        updated_date=COALESCE(?, updated_date),
                        venue=COALESCE(?, venue),
                        publication_url=COALESCE(?, publication_url),
                        pdf_url=COALESCE(?, pdf_url),
                        auto_analyze=CASE WHEN ? = 0 THEN 0 ELSE auto_analyze END,
                        abstract_source=COALESCE(?, abstract_source),
                        pdf_source=COALESCE(?, pdf_source),
                        fetched_at=?, metadata_json=?
                    WHERE id=?
                    """,
                    (
                        paper.openalex_id,
                        paper.doi,
                        paper.arxiv_id,
                        paper.title,
                        paper.normalized_title,
                        paper.abstract,
                        paper.publication_date,
                        paper.updated_date,
                        paper.venue,
                        paper.publication_url,
                        paper.pdf_url,
                        1 if auto_analyze else 0,
                        paper.abstract_source,
                        paper.pdf_source,
                        now,
                        metadata,
                        paper_id,
                    ),
                )
                is_new = False

            conn.execute("DELETE FROM paper_authors WHERE paper_id = ?", (paper_id,))
            for author_order, item in enumerate(paper.authors):
                tracked = (
                    conn.execute(
                        "SELECT id FROM authors WHERE openalex_id = ?", (item.openalex_id,)
                    ).fetchone()
                    if item.openalex_id
                    else None
                )
                conn.execute(
                    """
                    INSERT INTO paper_authors (
                        paper_id, tracked_author_id, openalex_author_id, name,
                        author_order, author_position, is_corresponding
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        paper_id,
                        int(tracked["id"]) if tracked else None,
                        item.openalex_id,
                        item.name,
                        author_order,
                        item.position,
                        1 if item.is_corresponding else 0,
                    ),
                )
            self._sync_paper_topics(conn, paper_id, paper.topics)
        return paper_id, is_new

    def set_local_pdf(self, paper_id: int, path: str) -> None:
        with self.transaction() as conn:
            conn.execute("UPDATE papers SET local_pdf_path = ? WHERE id = ?", (path, paper_id))

    def get_paper(self, paper_id: int) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()

    def list_papers(self, limit: int = 20) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT * FROM papers ORDER BY publication_date DESC, id DESC LIMIT ?", (limit,)
            ).fetchall()
        )

    def list_recent_papers_for_pdf_retry(self, since: str) -> list[sqlite3.Row]:
        """Return recent papers whose local PDF may need to be fetched again."""
        return list(
            self._conn.execute(
                """
                SELECT * FROM papers
                WHERE COALESCE(publication_date, discovered_at) >= ?
                ORDER BY COALESCE(publication_date, discovered_at) DESC, id DESC
                """,
                (since,),
            ).fetchall()
        )

    def dashboard_stats(self) -> dict[str, Any]:
        counts = self._conn.execute(
            """
            SELECT
                (SELECT count(*) FROM authors WHERE tracked = 1) AS authors,
                (SELECT count(*) FROM authors WHERE enabled = 1 AND tracked = 1) AS enabled_authors,
                (SELECT count(*) FROM authors WHERE tracked = 0) AS archived_authors,
                (SELECT count(*) FROM papers) AS papers,
                (SELECT count(*) FROM papers WHERE abstract IS NOT NULL) AS with_abstract,
                (SELECT count(*) FROM papers WHERE pdf_url IS NOT NULL OR local_pdf_path IS NOT NULL) AS with_pdf,
                (SELECT count(*) FROM analyses) AS analyses,
                (SELECT count(DISTINCT paper_id) FROM analyses) AS analyzed_papers
            """
        ).fetchone()
        kinds = self._conn.execute(
            """
            SELECT paper_kind, count(DISTINCT paper_id) AS count
            FROM analyses
            GROUP BY paper_kind
            ORDER BY count DESC, paper_kind
            """
        ).fetchall()
        years = self._conn.execute(
            """
            SELECT substr(publication_date, 1, 4) AS year, count(*) AS count
            FROM papers
            WHERE publication_date IS NOT NULL AND length(publication_date) >= 4
            GROUP BY year
            ORDER BY year DESC
            LIMIT 8
            """
        ).fetchall()
        return {
            **dict(counts or {}),
            "kinds": [dict(row) for row in kinds],
            "years": [dict(row) for row in years],
        }

    def search_papers(
        self,
        *,
        query: str = "",
        author_id: str = "",
        author_tags: Iterable[str] = (),
        topic_id: str = "",
        paper_kind: str = "",
        scope: str = "",
        has_pdf: bool | None = None,
        analyzed: bool | None = None,
        date_from: str = "",
        date_to: str = "",
        sort: str = "newest",
        limit: int | None = 25,
        offset: int = 0,
    ) -> tuple[list[sqlite3.Row], int]:
        where: list[str] = []
        params: list[Any] = []
        if query:
            where.append(
                "(p.title LIKE ? ESCAPE '\\' COLLATE NOCASE "
                "OR COALESCE(p.abstract, '') LIKE ? ESCAPE '\\' COLLATE NOCASE "
                "OR EXISTS (SELECT 1 FROM paper_authors paq WHERE paq.paper_id=p.id "
                "AND paq.name LIKE ? ESCAPE '\\' COLLATE NOCASE))"
            )
            escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            term = f"%{escaped}%"
            params.extend([term, term, term])
        if author_id:
            where.append(
                "EXISTS (SELECT 1 FROM paper_authors paa WHERE paa.paper_id=p.id "
                "AND paa.openalex_author_id=?)"
            )
            params.append(self._author_id(author_id))
        clean_author_tags = tuple(
            dict.fromkeys(
                " ".join(str(tag).strip().split())
                for tag in author_tags
                if str(tag).strip()
            )
        )
        if clean_author_tags:
            author_ids: list[str] = []
            tag_names: list[str] = []
            for value in clean_author_tags:
                candidate = short_openalex_id(value) or ""
                if candidate.startswith("A") and candidate[1:].isdigit():
                    author_ids.append(candidate)
                else:
                    tag_names.append(value)

            author_tag_clauses: list[str] = []
            author_tag_params: list[Any] = []
            if author_ids:
                placeholders = ",".join("?" for _ in author_ids)
                author_tag_clauses.append(
                    "EXISTS (SELECT 1 FROM paper_authors pat "
                    "WHERE pat.paper_id=p.id "
                    f"AND pat.openalex_author_id IN ({placeholders}))"
                )
                author_tag_params.extend(author_ids)
            if tag_names:
                placeholders = ",".join("?" for _ in tag_names)
                author_tag_clauses.append(
                    "EXISTS (SELECT 1 FROM paper_authors pat "
                    "JOIN author_tag_links atl ON atl.author_id=pat.tracked_author_id "
                    "JOIN author_tags atg ON atg.id=atl.tag_id "
                    "WHERE pat.paper_id=p.id "
                    f"AND atg.name COLLATE NOCASE IN ({placeholders}))"
                )
                author_tag_params.extend(tag_names)
            where.append("(" + " OR ".join(author_tag_clauses) + ")")
            params.extend(author_tag_params)
        if topic_id:
            where.append(
                "EXISTS (SELECT 1 FROM paper_topics ptt WHERE ptt.paper_id=p.id "
                "AND ptt.topic_id=?)"
            )
            params.append(self._topic_id(topic_id))
        if paper_kind:
            where.append(
                "EXISTS (SELECT 1 FROM analyses ak WHERE ak.paper_id=p.id AND ak.paper_kind=?)"
            )
            params.append(paper_kind)
        if scope:
            where.append(
                "EXISTS (SELECT 1 FROM analyses ascx WHERE ascx.paper_id=p.id AND ascx.scope=?)"
            )
            params.append(scope)
        if has_pdf is not None:
            where.append(
                "(p.pdf_url IS NOT NULL OR p.local_pdf_path IS NOT NULL)"
                if has_pdf
                else "(p.pdf_url IS NULL AND p.local_pdf_path IS NULL)"
            )
        if analyzed is not None:
            clause = "EXISTS" if analyzed else "NOT EXISTS"
            where.append(f"{clause} (SELECT 1 FROM analyses aa WHERE aa.paper_id=p.id)")
        if date_from:
            where.append("p.publication_date >= ?")
            params.append(date_from)
        if date_to:
            where.append("p.publication_date <= ?")
            params.append(date_to)

        where_sql = " WHERE " + " AND ".join(where) if where else ""
        order_map = {
            "newest": "COALESCE(p.publication_date, '') DESC, p.id DESC",
            "oldest": "COALESCE(p.publication_date, '9999-12-31') ASC, p.id ASC",
            "discovered": "p.discovered_at DESC, p.id DESC",
            "title": "p.title COLLATE NOCASE ASC, p.id ASC",
        }
        order_sql = order_map.get(sort, order_map["newest"])
        total = int(
            self._conn.execute(
                f"SELECT count(*) FROM papers p{where_sql}", tuple(params)
            ).fetchone()[0]
        )
        paging_sql = ""
        query_params: tuple[Any, ...] = tuple(params)
        if limit is not None:
            paging_sql = " LIMIT ? OFFSET ?"
            query_params = (*params, limit, offset)
        rows = self._conn.execute(
            f"""
            SELECT p.*,
                (SELECT group_concat(name, ' · ') FROM (
                    SELECT name FROM paper_authors
                    WHERE paper_id=p.id ORDER BY author_order
                )) AS author_names,
                COALESCE((
                    SELECT json_group_array(json_object(
                        'id', topic_id,
                        'name', display_name,
                        'score', score,
                        'is_primary', is_primary
                    ))
                    FROM (
                        SELECT pt.topic_id, t.display_name, pt.score, pt.is_primary
                        FROM paper_topics pt
                        JOIN topics t ON t.openalex_id=pt.topic_id
                        WHERE pt.paper_id=p.id
                        ORDER BY pt.topic_order
                    )
                ), '[]') AS topics_json,
                (SELECT count(*) FROM analyses ax WHERE ax.paper_id=p.id) AS analysis_count,
                (SELECT ax.scope FROM analyses ax WHERE ax.paper_id=p.id
                    ORDER BY CASE ax.scope WHEN 'fulltext' THEN 0 ELSE 1 END, ax.id DESC LIMIT 1
                ) AS latest_scope,
                (SELECT ax.paper_kind FROM analyses ax WHERE ax.paper_id=p.id
                    ORDER BY CASE ax.scope WHEN 'fulltext' THEN 0 ELSE 1 END, ax.id DESC LIMIT 1
                ) AS latest_kind,
                (SELECT ax.one_sentence_summary FROM analyses ax WHERE ax.paper_id=p.id
                    ORDER BY CASE ax.scope WHEN 'fulltext' THEN 0 ELSE 1 END, ax.id DESC LIMIT 1
                ) AS latest_summary
            FROM papers p
            {where_sql}
            ORDER BY {order_sql}{paging_sql}
            """,
            query_params,
        ).fetchall()
        return list(rows), total

    def paper_authors(self, paper_id: int) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                """
                SELECT pa.*, a.enabled AS tracked_enabled, a.tracked AS is_tracked
                FROM paper_authors pa
                LEFT JOIN authors a ON a.id = pa.tracked_author_id
                WHERE pa.paper_id=?
                ORDER BY pa.author_order
                """,
                (paper_id,),
            ).fetchall()
        )

    def paper_topics(self, paper_id: int) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                """
                SELECT t.*, pt.score, pt.is_primary, pt.topic_order
                FROM paper_topics pt
                JOIN topics t ON t.openalex_id=pt.topic_id
                WHERE pt.paper_id=?
                ORDER BY pt.topic_order, t.display_name COLLATE NOCASE
                """,
                (paper_id,),
            ).fetchall()
        )

    def list_topics(self) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                """
                SELECT t.*, count(pt.paper_id) AS paper_count
                FROM topics t
                JOIN paper_topics pt ON pt.topic_id=t.openalex_id
                GROUP BY t.openalex_id
                ORDER BY paper_count DESC, t.display_name COLLATE NOCASE
                """
            ).fetchall()
        )

    def get_paper_record(self, paper_id: int) -> PaperRecord | None:
        row = self.get_paper(paper_id)
        if row is None:
            return None
        metadata = json.loads(row["metadata_json"] or "{}")
        authors = [
            PaperAuthor(
                openalex_id=item["openalex_author_id"],
                name=item["name"],
                position=item["author_position"],
                is_corresponding=bool(item["is_corresponding"]),
            )
            for item in self.paper_authors(paper_id)
        ]
        topics = [
            PaperTopic(
                openalex_id=item["openalex_id"],
                name=item["display_name"],
                score=item["score"],
                is_primary=bool(item["is_primary"]),
                subfield_id=item["subfield_id"],
                subfield_name=item["subfield_name"],
                field_id=item["field_id"],
                field_name=item["field_name"],
                domain_id=item["domain_id"],
                domain_name=item["domain_name"],
                metadata=json.loads(item["metadata_json"] or "{}"),
            )
            for item in self.paper_topics(paper_id)
        ]
        return PaperRecord(
            openalex_id=row["openalex_id"],
            doi=row["doi"],
            arxiv_id=row["arxiv_id"],
            title=row["title"],
            abstract=row["abstract"],
            publication_date=row["publication_date"],
            updated_date=row["updated_date"],
            venue=row["venue"],
            publication_url=row["publication_url"],
            pdf_url=row["pdf_url"],
            abstract_source=row["abstract_source"],
            pdf_source=row["pdf_source"],
            authors=authors,
            topics=topics,
            metadata=metadata,
        )

    def paper_analyses(self, paper_id: int) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                """
                SELECT * FROM analyses WHERE paper_id=?
                ORDER BY CASE scope WHEN 'fulltext' THEN 0 ELSE 1 END, id DESC
                """,
                (paper_id,),
            ).fetchall()
        )

    def get_analysis_by_id(self, analysis_id: int) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM analyses WHERE id=?", (analysis_id,)
        ).fetchone()

    def get_run_errors(self, run_id: int) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT * FROM run_errors WHERE run_id=? ORDER BY id", (run_id,)
            ).fetchall()
        )

    def recent_errors(self, limit: int = 10) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                """
                SELECT e.*, r.started_at AS run_started_at
                FROM run_errors e JOIN runs r ON r.id=e.run_id
                ORDER BY e.id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        )

    def has_analysis(
        self,
        paper_id: int,
        scope: str,
        provider: str,
        model: str,
        prompt_version: str,
        content_sha256: str,
    ) -> bool:
        return (
            self.get_analysis(
                paper_id, scope, provider, model, prompt_version, content_sha256
            )
            is not None
        )

    def save_analysis(
        self,
        *,
        paper_id: int,
        scope: str,
        provider: str,
        model: str,
        prompt_version: str,
        content_sha256: str,
        input_chars: int,
        result: Any,
        analysis_strategy: str = "direct",
        context_window_tokens: int | None = None,
        chunk_count: int = 0,
        usage: dict[str, Any],
    ) -> int:
        payload = result.model_dump()
        list_fields = (
            "authors",
            "key_contributions",
            "assumptions",
            "methods_and_models",
            "main_results",
            "datasets",
            "supporting_evidence",
            "limitations_and_open_questions",
            "extended_outline",
            "appendix_summaries",
        )
        stored = {
            key: json_dumps(payload[key]) if key in list_fields else payload[key]
            for key in payload
        }
        columns = (
            "paper_id", "scope", "provider", "model", "prompt_version",
            "content_sha256", "created_at", "input_chars", "analysis_strategy",
            "context_window_tokens", "chunk_count", "name",
            "generated_by_model", "publication_date", "authors", "paper_kind",
            "source", "one_sentence_summary", "abstract_summary", "research_problem",
            "key_contributions", "assumptions", "definitions_and_notation",
            "methods_and_models", "main_results", "datasets", "supporting_evidence",
            "conclusion_summary", "limitations_and_open_questions", "extended_outline",
            "appendix_summaries", "notes", "result_json", "usage_json",
        )
        values = (
            paper_id, scope, provider, model, prompt_version, content_sha256, utcnow(),
            input_chars, analysis_strategy, context_window_tokens, chunk_count, stored["name"], stored["generated_by_model"],
            stored["publication_date"], stored["authors"], stored["paper_kind"],
            stored["source"], stored["one_sentence_summary"], stored["abstract_summary"],
            stored["research_problem"], stored["key_contributions"], stored["assumptions"],
            stored["definitions_and_notation"], stored["methods_and_models"],
            stored["main_results"], stored["datasets"], stored["supporting_evidence"],
            stored["conclusion_summary"], stored["limitations_and_open_questions"],
            stored["extended_outline"], stored["appendix_summaries"], stored["notes"],
            json_dumps(payload), json_dumps(usage),
        )
        placeholders = ", ".join("?" for _ in columns)
        with self.transaction() as conn:
            conn.execute(
                f"INSERT OR IGNORE INTO analyses ({', '.join(columns)}) VALUES ({placeholders})",
                values,
            )
            row = conn.execute(
                """
                SELECT id FROM analyses
                WHERE paper_id=? AND scope=? AND provider=? AND model=?
                  AND prompt_version=? AND content_sha256=?
                """,
                (paper_id, scope, provider, model, prompt_version, content_sha256),
            ).fetchone()
        assert row is not None
        return int(row["id"])

    def get_analysis(
        self,
        paper_id: int,
        scope: str,
        provider: str,
        model: str,
        prompt_version: str,
        content_sha256: str,
    ) -> sqlite3.Row | None:
        return self._conn.execute(
            """
            SELECT * FROM analyses
            WHERE paper_id=? AND scope=? AND provider=? AND model=?
              AND prompt_version=? AND content_sha256=?
            """,
            (paper_id, scope, provider, model, prompt_version, content_sha256),
        ).fetchone()

    def create_analysis_tasks(
        self,
        *,
        batch_id: str,
        paper_id: int,
        scope: str,
        profiles: list[tuple[str, str, str]],
        analysis_strategy: str = "direct",
        context_window_tokens: int | None = None,
    ) -> list[int]:
        now = utcnow()
        ids: list[int] = []
        with self.transaction() as conn:
            for profile, provider, model in profiles:
                cur = conn.execute(
                    """
                    INSERT INTO analysis_tasks (
                        batch_id, paper_id, scope, profile, provider, model, status, created_at,
                        analysis_strategy, context_window_tokens
                    ) VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?)
                    """,
                    (batch_id, paper_id, scope, profile, provider, model, now, analysis_strategy, context_window_tokens),
                )
                ids.append(int(cur.lastrowid))
        return ids

    def start_analysis_task(self, task_id: int) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE analysis_tasks SET status='running', started_at=?, error_text=NULL WHERE id=?",
                (utcnow(), task_id),
            )

    def finish_analysis_task(
        self, task_id: int, *, analysis_id: int | None = None, error: str | None = None,
        chunk_count: int = 0
    ) -> None:
        status = "failed" if error else "success"
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE analysis_tasks
                SET status=?, analysis_id=?, finished_at=?, error_text=?, chunk_count=?
                WHERE id=?
                """,
                (status, analysis_id, utcnow(), error, chunk_count, task_id),
            )

    def fail_incomplete_analysis_tasks(
        self, message: str, *, created_before: str | None = None
    ) -> int:
        where = "status IN ('queued', 'running')"
        params: list[Any] = [utcnow(), message]
        if created_before is not None:
            where += " AND created_at < ?"
            params.append(created_before)
        with self.transaction() as conn:
            cur = conn.execute(
                f"""
                UPDATE analysis_tasks
                SET status='failed', finished_at=?, error_text=?
                WHERE {where}
                """,
                tuple(params),
            )
        return cur.rowcount

    def analysis_tasks_for_paper(self, paper_id: int, limit: int = 30) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                """
                SELECT * FROM analysis_tasks
                WHERE paper_id=? ORDER BY id DESC LIMIT ?
                """,
                (paper_id, limit),
            ).fetchall()
        )

    def analysis_tasks_for_batch(self, batch_id: str) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT * FROM analysis_tasks WHERE batch_id=? ORDER BY id",
                (batch_id,),
            ).fetchall()
        )

    def active_analysis_tasks(self) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                """
                SELECT at.*, p.title AS paper_title
                FROM analysis_tasks at JOIN papers p ON p.id=at.paper_id
                WHERE at.status IN ('queued', 'running')
                ORDER BY at.id DESC
                """
            ).fetchall()
        )

    def visible_analysis_tasks_for_paper(
        self, paper_id: int, limit: int = 30
    ) -> list[sqlite3.Row]:
        """Return successful or currently active tasks, excluding inconclusive history."""
        return list(
            self._conn.execute(
                """
                SELECT * FROM analysis_tasks
                WHERE paper_id=? AND status IN ('queued', 'running', 'success')
                ORDER BY id DESC LIMIT ?
                """,
                (paper_id, limit),
            ).fetchall()
        )

    def list_authors_with_tags(
        self, *, tracked_only: bool = True, tag: str = "", query: str = ""
    ) -> list[sqlite3.Row]:
        where = ["a.tracked=1"] if tracked_only else []
        params: list[Any] = []
        if tag:
            where.append(
                "EXISTS (SELECT 1 FROM author_tag_links atl2 "
                "JOIN author_tags at2 ON at2.id=atl2.tag_id "
                "WHERE atl2.author_id=a.id AND at2.name=? COLLATE NOCASE)"
            )
            params.append(tag.strip())
        if query:
            escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            term = f"%{escaped}%"
            where.append(
                "(a.name LIKE ? ESCAPE '\\' COLLATE NOCASE "
                "OR COALESCE(a.institution, '') LIKE ? ESCAPE '\\' COLLATE NOCASE "
                "OR COALESCE(a.orcid, '') LIKE ? ESCAPE '\\' COLLATE NOCASE "
                "OR a.openalex_id LIKE ? ESCAPE '\\' COLLATE NOCASE "
                "OR EXISTS (SELECT 1 FROM author_tag_links atl3 "
                "JOIN author_tags at3 ON at3.id=atl3.tag_id "
                "WHERE atl3.author_id=a.id "
                "AND at3.name LIKE ? ESCAPE '\\' COLLATE NOCASE))"
            )
            params.extend([term, term, term, term, term])
        where_sql = " WHERE " + " AND ".join(where) if where else ""
        return list(
            self._conn.execute(
                f"""
                SELECT a.*,
                    COALESCE((SELECT group_concat(name, ', ') FROM (
                        SELECT at.name AS name
                        FROM author_tag_links atl
                        JOIN author_tags at ON at.id=atl.tag_id
                        WHERE atl.author_id=a.id
                        ORDER BY at.name COLLATE NOCASE
                    )), '') AS tags,
                    (SELECT count(DISTINCT pa.paper_id)
                     FROM paper_authors pa
                     WHERE pa.tracked_author_id=a.id) AS paper_count,
                    (SELECT count(*) FROM author_blogs ab
                     WHERE ab.author_id=a.id) AS blog_count
                FROM authors a
                {where_sql}
                ORDER BY a.name COLLATE NOCASE
                """,
                tuple(params),
            ).fetchall()
        )

    def list_tags(self) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                """
                SELECT at.id, at.name, count(a.id) AS author_count
                FROM author_tags at
                JOIN author_tag_links atl ON atl.tag_id=at.id
                JOIN authors a ON a.id=atl.author_id AND a.tracked=1
                GROUP BY at.id, at.name
                ORDER BY at.name COLLATE NOCASE
                """
            ).fetchall()
        )

    @staticmethod
    def _clean_tag(tag: str) -> str:
        cleaned = " ".join(tag.strip().split())
        if not cleaned:
            raise ValueError("Tag cannot be empty")
        if len(cleaned) > 40:
            raise ValueError("Tag must be at most 40 characters")
        return cleaned

    def add_author_tag(self, openalex_id: str, tag: str) -> bool:
        identifier = self._author_id(openalex_id)
        cleaned = self._clean_tag(tag)
        with self.transaction() as conn:
            author = conn.execute(
                "SELECT id FROM authors WHERE openalex_id=?", (identifier,)
            ).fetchone()
            if author is None:
                return False
            conn.execute(
                "INSERT OR IGNORE INTO author_tags(name, created_at) VALUES (?, ?)",
                (cleaned, utcnow()),
            )
            tag_row = conn.execute(
                "SELECT id FROM author_tags WHERE name=? COLLATE NOCASE", (cleaned,)
            ).fetchone()
            assert tag_row is not None
            conn.execute(
                "INSERT OR IGNORE INTO author_tag_links(author_id, tag_id) VALUES (?, ?)",
                (int(author["id"]), int(tag_row["id"])),
            )
        return True

    def remove_author_tag(self, openalex_id: str, tag: str) -> bool:
        identifier = self._author_id(openalex_id)
        cleaned = self._clean_tag(tag)
        with self.transaction() as conn:
            cur = conn.execute(
                """
                DELETE FROM author_tag_links
                WHERE author_id=(SELECT id FROM authors WHERE openalex_id=?)
                  AND tag_id=(SELECT id FROM author_tags WHERE name=? COLLATE NOCASE)
                """,
                (identifier, cleaned),
            )
            conn.execute(
                "DELETE FROM author_tags WHERE id NOT IN (SELECT tag_id FROM author_tag_links)"
            )
        return cur.rowcount > 0

    def get_analysis_chunk(
        self,
        *,
        paper_id: int,
        provider: str,
        model: str,
        prompt_version: str,
        content_sha256: str,
        chunk_index: int,
        context_window_tokens: int,
    ) -> sqlite3.Row | None:
        return self._conn.execute(
            """
            SELECT * FROM analysis_chunks
            WHERE paper_id=? AND provider=? AND model=? AND prompt_version=?
              AND content_sha256=? AND chunk_index=? AND context_window_tokens=?
            """,
            (
                paper_id,
                provider,
                model,
                prompt_version,
                content_sha256,
                chunk_index,
                context_window_tokens,
            ),
        ).fetchone()

    def save_analysis_chunk(
        self,
        *,
        paper_id: int,
        provider: str,
        model: str,
        prompt_version: str,
        content_sha256: str,
        chunk_index: int,
        context_window_tokens: int,
        summary: dict[str, Any],
        usage: dict[str, Any],
    ) -> int:
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO analysis_chunks(
                    paper_id, provider, model, prompt_version, content_sha256,
                    chunk_index, context_window_tokens, summary_json, usage_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper_id,
                    provider,
                    model,
                    prompt_version,
                    content_sha256,
                    chunk_index,
                    context_window_tokens,
                    json_dumps(summary),
                    json_dumps(usage),
                    utcnow(),
                ),
            )
            row = conn.execute(
                """
                SELECT id FROM analysis_chunks
                WHERE paper_id=? AND provider=? AND model=? AND prompt_version=?
                  AND content_sha256=? AND chunk_index=? AND context_window_tokens=?
                """,
                (
                    paper_id,
                    provider,
                    model,
                    prompt_version,
                    content_sha256,
                    chunk_index,
                    context_window_tokens,
                ),
            ).fetchone()
        assert row is not None
        return int(row["id"])

    @staticmethod
    def _query_execution_values(
        *,
        model_profiles: Iterable[str],
        context_window_tokens: int,
        paper_source_scope: str,
        pdf_strategy: str,
    ) -> tuple[str, int, str, str]:
        profiles = list(dict.fromkeys(
            value.strip() for value in model_profiles if value.strip()
        ))
        if not 512 <= context_window_tokens <= 1_000_000:
            raise ValueError("Context window must be between 512 and 1000000 tokens")
        if paper_source_scope not in {"auto", "abstract", "fulltext"}:
            raise ValueError("Paper source must be auto, abstract, or fulltext")
        if pdf_strategy not in {"auto", "direct", "chunked"}:
            raise ValueError("PDF strategy must be auto, direct, or chunked")
        if paper_source_scope == "abstract":
            pdf_strategy = "auto"
        return json_dumps(profiles), context_window_tokens, paper_source_scope, pdf_strategy

    def create_saved_query(
        self,
        *,
        name: str,
        prompt: str,
        source_type: str = "papers",
        search_query: str = "",
        auto_run_papers: bool = False,
        auto_run_blogs: bool = False,
        model_profiles: Iterable[str] = (),
        context_window_tokens: int = 16384,
        paper_source_scope: str = "auto",
        pdf_strategy: str = "chunked",
    ) -> int:
        name = name.strip()
        prompt = prompt.strip()
        if not name or not prompt:
            raise ValueError("Query name and prompt are required")
        profiles_json, context, scope, strategy = self._query_execution_values(
            model_profiles=model_profiles,
            context_window_tokens=context_window_tokens,
            paper_source_scope=paper_source_scope,
            pdf_strategy=pdf_strategy,
        )
        now = utcnow()
        with self.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO saved_queries(
                    name, source_type, search_query, prompt, is_builtin,
                    auto_run_papers, auto_run_blogs, model_profiles_json,
                    context_window_tokens, paper_source_scope, pdf_strategy,
                    created_at, updated_at
                ) VALUES (?, 'papers', '', ?, 0, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name[:120],
                    prompt[:10000],
                    1 if auto_run_papers else 0,
                    1 if auto_run_blogs else 0,
                    profiles_json,
                    context,
                    scope,
                    strategy,
                    now,
                    now,
                ),
            )
        return int(cur.lastrowid)

    def update_saved_query(
        self,
        query_id: int,
        *,
        name: str,
        prompt: str,
        source_type: str = "papers",
        search_query: str = "",
        auto_run_papers: bool | None = None,
        auto_run_blogs: bool | None = None,
        model_profiles: Iterable[str] = (),
        context_window_tokens: int = 16384,
        paper_source_scope: str = "auto",
        pdf_strategy: str = "chunked",
    ) -> bool:
        name = name.strip()
        prompt = prompt.strip()
        if not name or not prompt:
            raise ValueError("Query name and prompt are required")
        profiles_json, context, scope, strategy = self._query_execution_values(
            model_profiles=model_profiles,
            context_window_tokens=context_window_tokens,
            paper_source_scope=paper_source_scope,
            pdf_strategy=pdf_strategy,
        )
        with self.transaction() as conn:
            cur = conn.execute(
                """
                UPDATE saved_queries
                SET name=CASE WHEN is_builtin=1 THEN ? ELSE ? END,
                    source_type='papers', search_query='', prompt=?,
                    auto_run_papers=CASE WHEN ? IS NULL THEN auto_run_papers ELSE ? END,
                    auto_run_blogs=CASE WHEN ? IS NULL THEN auto_run_blogs ELSE ? END,
                    model_profiles_json=?, context_window_tokens=?,
                    paper_source_scope=?, pdf_strategy=?, updated_at=?
                WHERE id=?
                """,
                (
                    DEFAULT_SUMMARY_QUERY_NAME,
                    name[:120],
                    prompt[:10000],
                    None if auto_run_papers is None else int(auto_run_papers),
                    0 if auto_run_papers is None else int(auto_run_papers),
                    None if auto_run_blogs is None else int(auto_run_blogs),
                    0 if auto_run_blogs is None else int(auto_run_blogs),
                    profiles_json,
                    context,
                    scope,
                    strategy,
                    utcnow(),
                    query_id,
                ),
            )
        return cur.rowcount > 0

    def update_automatic_queries(
        self,
        query_ids: Iterable[int],
        *,
        paper_query_ids: Iterable[int],
        blog_query_ids: Iterable[int],
    ) -> None:
        all_ids = list(dict.fromkeys(int(value) for value in query_ids))
        paper_ids = {int(value) for value in paper_query_ids}
        blog_ids = {int(value) for value in blog_query_ids}
        now = utcnow()
        with self.transaction() as conn:
            for query_id in all_ids:
                conn.execute(
                    """
                    UPDATE saved_queries
                    SET auto_run_papers=?, auto_run_blogs=?, updated_at=?
                    WHERE id=?
                    """,
                    (
                        1 if query_id in paper_ids else 0,
                        1 if query_id in blog_ids else 0,
                        now,
                        query_id,
                    ),
                )

    def delete_saved_query(self, query_id: int) -> bool:
        with self.transaction() as conn:
            cur = conn.execute(
                "DELETE FROM saved_queries WHERE id=? AND is_builtin=0", (query_id,)
            )
        return cur.rowcount > 0

    def get_saved_query(self, query_id: int) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM saved_queries WHERE id=?", (query_id,)).fetchone()

    def list_saved_queries(self) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                """
                SELECT q.*,
                    (SELECT count(*) FROM query_runs qr WHERE qr.query_id=q.id) AS run_count,
                    (SELECT qr.status FROM query_runs qr WHERE qr.query_id=q.id ORDER BY qr.id DESC LIMIT 1) AS latest_status
                FROM saved_queries q
                ORDER BY q.is_builtin DESC, q.updated_at DESC, q.id DESC
                """
            ).fetchall()
        )

    def list_automatic_queries(self, content_type: str) -> list[sqlite3.Row]:
        if content_type not in {"papers", "blogs"}:
            raise ValueError("Content type must be papers or blogs")
        column = "auto_run_papers" if content_type == "papers" else "auto_run_blogs"
        return list(
            self._conn.execute(
                f"SELECT * FROM saved_queries WHERE {column}=1 ORDER BY id"
            ).fetchall()
        )

    def create_query_runs(
        self,
        *,
        batch_id: str,
        query_id: int,
        profiles: list[tuple[str, str, str]],
        context_window_tokens: int,
        selection: dict[str, Any] | None = None,
    ) -> list[int]:
        now = utcnow()
        selection_json = json_dumps(selection or {})
        ids: list[int] = []
        with self.transaction() as conn:
            for profile, provider, model in profiles:
                cur = conn.execute(
                    """
                    INSERT INTO query_runs(
                        batch_id, query_id, profile, provider, model, status,
                        context_window_tokens, selection_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?)
                    """,
                    (
                        batch_id,
                        query_id,
                        profile,
                        provider,
                        model,
                        context_window_tokens,
                        selection_json,
                        now,
                    ),
                )
                ids.append(int(cur.lastrowid))
        return ids

    def start_query_run(self, run_id: int) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE query_runs SET status='running', started_at=?, error_text=NULL WHERE id=?",
                (utcnow(), run_id),
            )

    def finish_query_run(
        self,
        run_id: int,
        *,
        matched_count: int,
        input_chars: int,
        output: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE query_runs
                SET status=?, matched_count=?, input_chars=?, output_json=?,
                    finished_at=?, error_text=?
                WHERE id=?
                """,
                (
                    "failed" if error else "success",
                    matched_count,
                    input_chars,
                    json_dumps(output) if output is not None else None,
                    utcnow(),
                    error,
                    run_id,
                ),
            )
            conn.execute(
                """
                UPDATE saved_queries SET last_run_at=?
                WHERE id=(SELECT query_id FROM query_runs WHERE id=?)
                """,
                (utcnow(), run_id),
            )

    def query_runs_for_batch(self, batch_id: str) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT * FROM query_runs WHERE batch_id=? ORDER BY id", (batch_id,)
            ).fetchall()
        )

    def active_query_runs(self) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                """
                SELECT qr.*, q.name AS query_name
                FROM query_runs qr JOIN saved_queries q ON q.id=qr.query_id
                WHERE qr.status IN ('queued', 'running')
                ORDER BY qr.id DESC
                """
            ).fetchall()
        )

    def list_query_runs(self, query_id: int | None = None, limit: int = 100) -> list[sqlite3.Row]:
        if query_id is None:
            return list(
                self._conn.execute(
                    """
                    SELECT qr.*, q.name AS query_name
                    FROM query_runs qr JOIN saved_queries q ON q.id=qr.query_id
                    ORDER BY qr.id DESC LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            )
        return list(
            self._conn.execute(
                """
                SELECT qr.*, q.name AS query_name
                FROM query_runs qr JOIN saved_queries q ON q.id=qr.query_id
                WHERE qr.query_id=? ORDER BY qr.id DESC LIMIT ?
                """,
                (query_id, limit),
            ).fetchall()
        )

    def query_runs_for_paper(
        self, paper_id: int, *, successful_only: bool = True, limit: int = 100
    ) -> list[sqlite3.Row]:
        """Return saved-prompt runs explicitly scoped to one paper."""
        rows = self.list_query_runs(limit=max(limit * 5, 100))
        matched: list[sqlite3.Row] = []
        for row in rows:
            if successful_only and row["status"] != "success":
                continue
            try:
                selection = json.loads(row["selection_json"] or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            paper_ids = selection.get("paper_ids") if isinstance(selection, dict) else None
            if not isinstance(paper_ids, list):
                continue
            try:
                ids = {int(value) for value in paper_ids}
            except (TypeError, ValueError):
                continue
            if paper_id in ids:
                matched.append(row)
                if len(matched) >= limit:
                    break
        return matched

    def query_runs_for_blog_entry(
        self, entry_id: int, *, successful_only: bool = True, limit: int = 100
    ) -> list[sqlite3.Row]:
        rows = self.list_query_runs(limit=max(limit * 5, 100))
        matched: list[sqlite3.Row] = []
        for row in rows:
            if successful_only and row["status"] != "success":
                continue
            try:
                selection = json.loads(row["selection_json"] or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            entry_ids = selection.get("blog_entry_ids") if isinstance(selection, dict) else None
            if not isinstance(entry_ids, list):
                continue
            try:
                ids = {int(value) for value in entry_ids}
            except (TypeError, ValueError):
                continue
            if entry_id in ids:
                matched.append(row)
                if len(matched) >= limit:
                    break
        return matched

    def fail_incomplete_query_runs(self, message: str) -> int:
        with self.transaction() as conn:
            cur = conn.execute(
                """
                UPDATE query_runs SET status='failed', finished_at=?, error_text=?
                WHERE status IN ('queued', 'running')
                """,
                (utcnow(), message),
            )
        return cur.rowcount


    def start_run(self) -> int:
        with self.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO runs(started_at, status) VALUES (?, 'running')", (utcnow(),)
            )
        return int(cur.lastrowid)

    def finish_run(
        self,
        run_id: int,
        status: str,
        stats: dict[str, int],
        error_text: str | None = None,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE runs SET finished_at=?, status=?, authors_checked=?,
                    papers_found=?, papers_analyzed=?, pdfs_downloaded=?,
                    blogs_checked=?, blog_updates_found=?, errors_count=?, error_text=?
                WHERE id=?
                """,
                (
                    utcnow(),
                    status,
                    stats.get("authors_checked", 0),
                    stats.get("papers_found", 0),
                    stats.get("papers_analyzed", 0),
                    stats.get("pdfs_downloaded", 0),
                    stats.get("blogs_checked", 0),
                    stats.get("blog_updates_found", 0),
                    stats.get("errors_count", 0),
                    error_text,
                    run_id,
                ),
            )

    def record_error(
        self,
        run_id: int,
        *,
        stage: str,
        message: str,
        author_openalex_id: str | None = None,
        paper_identity_key: str | None = None,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO run_errors (
                    run_id, author_openalex_id, paper_identity_key,
                    stage, message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    author_openalex_id,
                    paper_identity_key,
                    stage,
                    message,
                    utcnow(),
                ),
            )

    def get_run(self, run_id: int) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()

    def list_runs(self, limit: int = 20) -> list[sqlite3.Row]:
        return list(
            self._conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        )
