from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from paperwatch.analysis_jobs import AnalysisBatchSnapshot
from paperwatch.config import load_config
from paperwatch.db import Database
from paperwatch.query_jobs import QueryBatchSnapshot
from paperwatch.web import RunSnapshot, create_app


class RunStub:
    def snapshot(self):
        return RunSnapshot()


class AnalysisStub:
    def snapshot(self):
        return AnalysisBatchSnapshot()


class QueryStub:
    def snapshot(self):
        return QueryBatchSnapshot()


class AuthorClient:
    def close(self):
        return None


def web_client(config_path: Path) -> TestClient:
    return TestClient(
        create_app(
            config_path,
            run_controller=RunStub(),
            analysis_controller=AnalysisStub(),
            query_controller=QueryStub(),
            author_client_factory=lambda config: AuthorClient(),
        )
    )


def test_content_list_uses_lightweight_rows_and_bounded_selected_preview(
    config_path, author_candidate, paper_record
):
    config = load_config(config_path)
    huge_text = "article body " * 10000
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        paper_id, _ = db.upsert_paper(paper_record)
        blog_id = db.add_author_blog(author_candidate.openalex_id, "https://example.test/blog")
        for index in range(30):
            entry_id = db.save_blog_entry(
                blog_id=blog_id,
                title=f"Large entry {index:02d}",
                content_sha256=f"large-{index}",
                normalized_text=huge_text,
                rendered_html=f"<p>{huge_text}</p>",
                diff_text=huge_text,
                change_kind="changed",
                previous_entry_id=None,
                entry_kind="rss",
                external_id=f"large-{index}",
                source_url=f"https://example.test/blog/{index}",
            )
            db.set_content_selected("blog", entry_id, True)

        blog_rows = db.list_blog_entries(limit=1, table_view=True)
        assert "normalized_text" not in blog_rows[0].keys()
        assert "rendered_html" not in blog_rows[0].keys()
        assert "diff_text" not in blog_rows[0].keys()

        paper_rows, total = db.search_papers(limit=1, table_view=True)
        assert total == 1
        assert paper_rows[0]["id"] == paper_id
        assert "abstract" not in paper_rows[0].keys()
        assert "metadata_json" not in paper_rows[0].keys()

    response = web_client(config_path).get("/")
    assert response.status_code == 200
    assert 'name="page_size" value="250"' in response.text
    assert "Showing the newest 25 selected records" in response.text
    assert response.text.count('class="content-row content-row-blog') == 25


def test_large_search_selects_are_rendered_lazily():
    script = Path("src/paperwatch/web_static/app.js").read_text(encoding="utf-8")
    assert "const maxRenderedOptions = 80" in script
    assert "Keep typing to narrow the list" in script
    assert "list.replaceChildren(fragment)" in script


def test_profile_controls_select_actual_controls_not_model_options():
    """Model option metadata must not be mistaken for the PDF strategy control.

    A broad ``[data-pdf-strategy]`` selector matches the first model ``option``.
    Dispatching its change event bubbles to the model picker and recursively calls
    profile synchronization until the browser tab is CPU-bound.
    """
    script = Path("src/paperwatch/web_static/app.js").read_text(encoding="utf-8")
    assert 'data-paper-processing' in script
    assert 'data-context-window' in script
    assert 'data-model-picker' in script
    assert 'select[data-source-scope]' not in script
    assert 'select[data-pdf-strategy]' not in script


def test_filtered_select_all_is_set_based(
    config_path, author_candidate, paper_record
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        blog_id = db.add_author_blog(
            author_candidate.openalex_id, "https://example.test/bulk"
        )
        for index in range(40):
            db.upsert_paper(
                replace(
                    paper_record,
                    openalex_id=f"W{1000 + index}",
                    doi=f"10.1000/bulk-{index}",
                    arxiv_id=None,
                    title=f"Bulk match paper {index:02d}",
                )
            )
            db.save_blog_entry(
                blog_id=blog_id,
                title=f"Bulk match blog {index:02d}",
                content_sha256=f"bulk-{index}",
                normalized_text="body",
                rendered_html=None,
                diff_text="",
                change_kind="changed",
                previous_entry_id=None,
                entry_kind="rss",
                external_id=f"bulk-{index}",
                source_url=f"https://example.test/bulk/{index}",
            )
        db.upsert_paper(
            replace(
                paper_record,
                openalex_id="W9999",
                doi="10.1000/unmatched",
                arxiv_id=None,
                title="Unmatched paper",
            )
        )
        db.save_blog_entry(
            blog_id=blog_id,
            title="Unmatched blog",
            content_sha256="unmatched",
            normalized_text="body",
            rendered_html=None,
            diff_text="",
            change_kind="changed",
            previous_entry_id=None,
            entry_kind="rss",
            external_id="unmatched",
            source_url="https://example.test/bulk/unmatched",
        )

        previous_limit = db._conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 12)
        try:
            changed = db.bulk_set_filtered_content_selected(
                selected=True,
                paper_filters={"title_query": "Bulk match", "selected": False},
                blog_filters={"title_query": "Bulk match", "selected": False},
            )
        finally:
            db._conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, previous_limit)

        assert changed == 80
        assert db._conn.execute(
            "SELECT count(*) FROM papers WHERE selected=1"
        ).fetchone()[0] == 40
        assert db._conn.execute(
            "SELECT count(*) FROM blog_entries WHERE selected=1"
        ).fetchone()[0] == 40
        assert db._conn.execute(
            "SELECT selected FROM papers WHERE title='Unmatched paper'"
        ).fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT selected FROM blog_entries WHERE title='Unmatched blog'"
        ).fetchone()[0] == 0


def test_select_results_endpoint_does_not_fetch_matching_ids(
    config_path, author_candidate, paper_record, monkeypatch
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        paper_id, _ = db.upsert_paper(paper_record)

    original_search = Database.search_papers
    original_list = Database.list_blog_entries

    def search_without_id_materialization(self, **kwargs):
        assert not kwargs.get("ids_only")
        return original_search(self, **kwargs)

    def list_without_id_materialization(self, limit=100, **kwargs):
        assert not kwargs.get("ids_only")
        return original_list(self, limit=limit, **kwargs)

    monkeypatch.setattr(Database, "search_papers", search_without_id_materialization)
    monkeypatch.setattr(Database, "list_blog_entries", list_without_id_materialization)

    response = web_client(config_path).post(
        "/content/select-results",
        data={"content": "papers", "selected": "1", "title_query": "Minimal Agents"},
        headers={"X-Paperwatch-Async": "1"},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True, "selected": True, "changed": 1}
    with Database(config.paths.database) as db:
        assert db.get_paper(paper_id)["selected"] == 1


def test_unselect_all_returns_selected_results_to_the_results_table(
    config_path, author_candidate, paper_record
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        db.upsert_paper(paper_record)

    client = web_client(config_path)
    response = client.post(
        "/content/select-results",
        data={"content": "papers", "selected": "1", "title_query": "Minimal Agents"},
        headers={"X-Paperwatch-Async": "1"},
    )
    assert response.status_code == 200

    selected_page = client.get("/?content=papers&title_query=Minimal+Agents")
    selected_body = selected_page.text.split("<tbody data-selected-content-body>", 1)[1].split(
        "</tbody>", 1
    )[0]
    results_body = selected_page.text.split('<tbody id="paper-table-body">', 1)[1].split(
        "</tbody>", 1
    )[0]
    assert paper_record.title in selected_body
    assert paper_record.title not in results_body

    response = client.post(
        "/content/selected/unselect",
        headers={"X-Paperwatch-Async": "1"},
    )
    assert response.status_code == 200
    assert response.json()["changed"] == 1

    refreshed_page = client.get("/?content=papers&title_query=Minimal+Agents")
    selected_body = refreshed_page.text.split("<tbody data-selected-content-body>", 1)[1].split(
        "</tbody>", 1
    )[0]
    results_body = refreshed_page.text.split('<tbody id="paper-table-body">', 1)[1].split(
        "</tbody>", 1
    )[0]
    assert paper_record.title not in selected_body
    assert paper_record.title in results_body


def test_content_blog_summary_lookup_is_not_correlated_per_row(
    config_path, author_candidate
):
    """A large query history must be expanded once, not once per visible row."""
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        blog_id = db.add_author_blog(
            author_candidate.openalex_id, "https://example.test/performance-blog"
        )
        for index in range(40):
            db.save_blog_entry(
                blog_id=blog_id,
                title=f"Performance entry {index:02d}",
                content_sha256=f"performance-{index}",
                normalized_text="body",
                rendered_html="<p>body</p>",
                diff_text="",
                change_kind="changed",
                previous_entry_id=None,
                entry_kind="rss",
                external_id=f"performance-{index}",
                source_url=f"https://example.test/performance-blog/{index}",
            )
        query_id = db.create_saved_query(name="Performance summary", prompt="Summarize")
        now = "2026-08-04T12:00:00+00:00"
        db._conn.executemany(
            """
            INSERT INTO query_runs(
                batch_id, query_id, profile, provider, model, status,
                context_window_tokens, selection_json, created_at, started_at, finished_at
            ) VALUES (?, ?, ?, 'ollama', 'model', 'success', 2048, ?, ?, ?, ?)
            """,
            (
                (
                    f"performance-{index}",
                    query_id,
                    f"profile-{index}",
                    json.dumps(
                        {"blog_entry_ids": [1_000_000 + value for value in range(40)]}
                    ),
                    now,
                    now,
                    now,
                )
                for index in range(200)
            ),
        )
        db._conn.commit()

        progress_calls = 0

        def stop_runaway_query() -> int:
            nonlocal progress_calls
            progress_calls += 1
            return 1 if progress_calls > 500 else 0

        db._conn.set_progress_handler(stop_runaway_query, 1_000)
        try:
            rows = db.list_blog_entries(limit=25, table_view=True)
        finally:
            db._conn.set_progress_handler(None, 0)

        assert len(rows) == 25
        assert all(int(row["has_summary"]) == 0 for row in rows)
        assert progress_calls < 500


def test_bulk_blog_delete_is_bounded_and_empty_content_skips_blog_row_queries(
    config_path, author_candidate, paper_record, monkeypatch
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        db.upsert_paper(paper_record)
        blog_id = db.add_author_blog(
            author_candidate.openalex_id, "https://example.test/delete-performance"
        )
        now = "2026-08-04T12:00:00+00:00"
        db._conn.executemany(
            """
            INSERT INTO blog_entries(
                id, blog_id, previous_entry_id, title, content_sha256,
                normalized_text, diff_text, change_kind, entry_kind,
                external_id, source_url, fetched_at
            ) VALUES (?, ?, ?, ?, ?, 'body', '', 'changed', 'rss', ?, ?, ?)
            """,
            (
                (
                    index,
                    blog_id,
                    index - 1 if index > 1 else None,
                    f"Entry {index}",
                    f"hash-{index}",
                    f"entry-{index}",
                    f"https://example.test/delete-performance/{index}",
                    now,
                )
                for index in range(1, 3_001)
            ),
        )
        # Simulate an existing database created before the performance index.
        db._conn.execute("DROP INDEX idx_blog_entries_previous")
        db._conn.execute("DROP INDEX idx_blog_entries_blog")
        db._conn.commit()

    with Database(config.paths.database) as db:
        indexes = {
            str(row["name"])
            for row in db._conn.execute("PRAGMA index_list(blog_entries)").fetchall()
        }
        assert "idx_blog_entries_previous" in indexes
        assert "idx_blog_entries_blog" in indexes

        progress_calls = 0

        def count_steps() -> int:
            nonlocal progress_calls
            progress_calls += 1
            return 0

        db._conn.set_progress_handler(count_steps, 1_000)
        try:
            assert db.delete_all_blog_entries() == 3_000
        finally:
            db._conn.set_progress_handler(None, 0)
        assert progress_calls < 1_000
        assert db.count_blog_entries() == 0

    def unexpected_blog_rows(*args, **kwargs):
        raise AssertionError("empty Content pages must not build blog row metadata")

    monkeypatch.setattr(Database, "list_blog_entries", unexpected_blog_rows)
    client = web_client(config_path)
    assert client.get("/").status_code == 200
    assert client.get("/blogs").status_code == 200


def test_empty_blog_content_skips_expensive_saved_prompt_filters(
    config_path, author_candidate, monkeypatch
):
    """A remembered analyzed filter must not scan query history after bulk delete."""
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        blog_id = db.add_author_blog(
            author_candidate.openalex_id, "https://example.test/empty-history"
        )
        db.save_blog_entry(
            blog_id=blog_id,
            title="Deleted entry",
            content_sha256="deleted-entry",
            normalized_text="body",
            diff_text="",
            change_kind="changed",
            previous_entry_id=None,
            entry_kind="rss",
            external_id="deleted-entry",
            source_url="https://example.test/empty-history/entry",
        )
        query_id = db.create_saved_query(name="Historical prompt", prompt="Summarize")
        now = "2026-08-04T12:00:00+00:00"
        selection = json.dumps({"blog_entry_ids": list(range(1, 2_001))})
        db._conn.executemany(
            """
            INSERT INTO query_runs(
                batch_id, query_id, profile, provider, model, status,
                context_window_tokens, selection_json, created_at, started_at, finished_at
            ) VALUES (?, ?, ?, 'ollama', 'model', 'success', 2048, ?, ?, ?, ?)
            """,
            (
                (f"historical-{index}", query_id, f"profile-{index}", selection, now, now, now)
                for index in range(100)
            ),
        )
        db._conn.commit()
        assert db.delete_all_blog_entries() == 1
        assert not db.has_visible_blog_entries()

    def unexpected_filtered_blog_query(*args, **kwargs):
        raise AssertionError(
            "empty Content pages must not evaluate blog filters or saved-prompt history"
        )

    monkeypatch.setattr(Database, "count_blog_entries", unexpected_filtered_blog_query)
    monkeypatch.setattr(Database, "list_blog_entries", unexpected_filtered_blog_query)

    response = web_client(config_path).get("/?content=blogs&analyzed=yes")
    assert response.status_code == 200
    assert "0 matching records" in response.text
    assert "No downloaded content matches this search." in response.text
