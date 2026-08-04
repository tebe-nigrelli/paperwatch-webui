from __future__ import annotations

import pytest

from paperwatch.db import Database
from paperwatch.models import stable_hash


def test_author_lifecycle(tmp_path, author_candidate):
    with Database(tmp_path / "db.sqlite") as db:
        author_id = db.add_author(author_candidate)
        assert author_id > 0
        assert db.get_author("A123")["name"] == "Ada Lovelace"
        assert len(db.list_authors(enabled_only=True)) == 1
        assert db.set_author_enabled("A123", False)
        assert db.list_authors(enabled_only=True) == []
        assert db.set_author_enabled("A123", True)
        db.mark_author_checked("A123", "2026-07-24T00:00:00+00:00")
        assert db.get_author("A123")["last_checked_at"].startswith("2026-07-24")
        assert db.remove_author("A123")
        archived = db.get_author("A123")
        assert archived is not None
        assert archived["tracked"] == 0
        assert archived["enabled"] == 0
        assert db.list_authors(tracked_only=True) == []
        linked = db._conn.execute(
            "SELECT tracked_author_id FROM paper_authors WHERE openalex_author_id='A123'"
        ).fetchone()
        if linked is not None:
            assert linked[0] == author_id
        db.add_author(author_candidate)
        assert db.get_author("A123")["tracked"] == 1


def test_paper_upsert_deduplicates_and_links_authors(tmp_path, author_candidate, paper_record):
    with Database(tmp_path / "db.sqlite") as db:
        db.add_author(author_candidate)
        paper_id, is_new = db.upsert_paper(paper_record)
        same_id, is_new_again = db.upsert_paper(paper_record)
        assert is_new is True
        assert is_new_again is False
        assert same_id == paper_id
        assert len(db.list_papers()) == 1
        rows = db._conn.execute("SELECT * FROM paper_authors WHERE paper_id=?", (paper_id,)).fetchall()
        assert len(rows) == 2
        tracked = [row for row in rows if row["openalex_author_id"] == "A123"][0]
        assert tracked["tracked_author_id"] is not None
        assert db.list_authors_with_tags()[0]["paper_count"] == 1


def test_analysis_cache_and_run_accounting(tmp_path, paper_record, analysis_result):
    with Database(tmp_path / "db.sqlite") as db:
        paper_id, _ = db.upsert_paper(paper_record)
        digest = stable_hash("prompt")
        assert not db.has_analysis(paper_id, "abstract", "ollama", "m", "v1", digest)
        analysis_id = db.save_analysis(
            paper_id=paper_id,
            scope="abstract",
            provider="ollama",
            model="m",
            prompt_version="v1",
            content_sha256=digest,
            input_chars=12,
            result=analysis_result,
            usage={"eval_count": 10},
        )
        assert analysis_id > 0
        assert db.has_analysis(paper_id, "abstract", "ollama", "m", "v1", digest)
        assert db.get_analysis(paper_id, "abstract", "ollama", "m", "v1", digest)["one_sentence_summary"]

        run_id = db.start_run()
        db.record_error(run_id, stage="test", message="boom", author_openalex_id="A123")
        db.finish_run(
            run_id,
            "partial",
            {
                "authors_checked": 1,
                "papers_found": 1,
                "papers_analyzed": 1,
                "pdfs_downloaded": 0,
                "errors_count": 1,
            },
        )
        run = db.get_run(run_id)
        assert run["status"] == "partial"
        assert run["errors_count"] == 1
        assert db._conn.execute("SELECT count(*) FROM run_errors").fetchone()[0] == 1


def test_readding_author_enables_and_full_urls_are_accepted(tmp_path, author_candidate):
    with Database(tmp_path / "db.sqlite") as db:
        db.add_author(author_candidate)
        assert db.set_author_enabled("https://openalex.org/A123", False)
        db.add_author(author_candidate)
        assert db.get_author("https://openalex.org/a123")["enabled"] == 1
        assert db.remove_author("https://openalex.org/A123")


def test_paper_update_replaces_stale_authors_and_keeps_identity(tmp_path, paper_record):
    from copy import deepcopy
    from paperwatch.models import PaperAuthor

    with Database(tmp_path / "db.sqlite") as db:
        paper_id, _ = db.upsert_paper(paper_record)
        original_identity = db.get_paper(paper_id)["identity_key"]
        updated = deepcopy(paper_record)
        updated.doi = "10.1000/new-doi"
        updated.authors = [PaperAuthor("A777", "Grace Hopper", "first", True)]
        same_id, is_new = db.upsert_paper(updated)
        assert same_id == paper_id and not is_new
        assert db.get_paper(paper_id)["identity_key"] == original_identity
        rows = db._conn.execute("SELECT * FROM paper_authors WHERE paper_id=?", (paper_id,)).fetchall()
        assert [(row["name"], row["author_position"]) for row in rows] == [("Grace Hopper", "first")]


def test_paper_deduplicates_by_doi_arxiv_and_title(tmp_path, paper_record):
    from dataclasses import replace

    with Database(tmp_path / "db.sqlite") as db:
        paper_id, _ = db.upsert_paper(paper_record)
        by_doi = replace(paper_record, openalex_id="W999", arxiv_id=None)
        assert db.upsert_paper(by_doi)[0] == paper_id

        second = replace(
            paper_record,
            openalex_id="W888",
            doi=None,
            arxiv_id="2607.99999",
            title="Another Work",
        )
        second_id, _ = db.upsert_paper(second)
        by_arxiv = replace(second, openalex_id="W777")
        assert db.upsert_paper(by_arxiv)[0] == second_id

        title_only = replace(
            paper_record,
            openalex_id=None,
            doi=None,
            arxiv_id=None,
            title="Title Only",
            publication_date="2025-01-02",
        )
        title_id, _ = db.upsert_paper(title_only)
        later_identified = replace(title_only, openalex_id="W12345")
        assert db.upsert_paper(later_identified)[0] == title_id


def test_transaction_rolls_back_and_newer_schema_is_rejected(tmp_path):
    import sqlite3
    from paperwatch.db import DatabaseError, SCHEMA_VERSION

    path = tmp_path / "db.sqlite"
    with Database(path) as db:
        with pytest.raises(RuntimeError):
            with db.transaction() as conn:
                conn.execute("INSERT INTO runs(started_at, status) VALUES ('x', 'running')")
                raise RuntimeError("rollback")
        assert db._conn.execute("SELECT count(*) FROM runs").fetchone()[0] == 0

    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    conn.close()
    with pytest.raises(DatabaseError, match="newer"):
        Database(path)


def test_adding_author_relinks_existing_coauthorship(tmp_path, author_candidate, paper_record):
    with Database(tmp_path / "db.sqlite") as db:
        paper_id, _ = db.upsert_paper(paper_record)
        before = db._conn.execute(
            "SELECT tracked_author_id FROM paper_authors WHERE paper_id=? AND openalex_author_id='A123'",
            (paper_id,),
        ).fetchone()[0]
        assert before is None
        author_id = db.add_author(author_candidate)
        after = db._conn.execute(
            "SELECT tracked_author_id FROM paper_authors WHERE paper_id=? AND openalex_author_id='A123'",
            (paper_id,),
        ).fetchone()[0]
        assert after == author_id


def test_version_zero_database_is_upgraded_without_data_loss(tmp_path):
    import sqlite3
    from paperwatch.db import SCHEMA, SCHEMA_VERSION

    path = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.execute(
        """
        INSERT INTO authors(openalex_id, name, enabled, added_at, metadata_json)
        VALUES ('A123', 'Legacy Author', 1, '2026-01-01T00:00:00+00:00', '{}')
        """
    )
    conn.execute("PRAGMA user_version = 0")
    conn.commit()
    conn.close()

    with Database(path) as db:
        assert db.get_author("A123")["name"] == "Legacy Author"
        assert db._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert db._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_analysis_fields_are_named_columns_and_round_trip(tmp_path, paper_record, analysis_result):
    import json

    with Database(tmp_path / "analysis-columns.sqlite") as db:
        paper_id, _ = db.upsert_paper(paper_record)
        digest = stable_hash("structured prompt")
        db.save_analysis(
            paper_id=paper_id,
            scope="fulltext",
            provider="ollama",
            model="test-model",
            prompt_version="v2",
            content_sha256=digest,
            input_chars=1234,
            result=analysis_result,
            usage={"eval_count": 20},
        )
        row = db.get_analysis(
            paper_id, "fulltext", "ollama", "test-model", "v2", digest
        )
        assert row is not None
        expected_columns = set(type(analysis_result).model_fields)
        assert expected_columns.issubset(set(row.keys()))
        assert row["name"] == analysis_result.name
        assert json.loads(row["authors"]) == analysis_result.authors
        assert json.loads(row["methods_and_models"])[0]["kind"] == "model"
        assert json.loads(row["main_results"])[0]["result_type"] == "empirical_finding"
        assert json.loads(row["result_json"]) == analysis_result.model_dump()
        assert json.loads(row["usage_json"])["eval_count"] == 20


def test_v1_analysis_schema_is_archived_and_replaced(tmp_path):
    import sqlite3
    from paperwatch.db import ANALYSES_SCHEMA, COMMON_SCHEMA, SCHEMA_VERSION

    path = tmp_path / "v1.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(COMMON_SCHEMA)
    conn.executescript(
        """
        CREATE TABLE analyses (
            id INTEGER PRIMARY KEY,
            paper_id INTEGER NOT NULL,
            scope TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL,
            input_chars INTEGER NOT NULL,
            summary TEXT NOT NULL,
            relevance_score REAL NOT NULL,
            novelty_score REAL NOT NULL,
            confidence REAL NOT NULL,
            result_json TEXT NOT NULL,
            usage_json TEXT NOT NULL DEFAULT '{}'
        );
        INSERT INTO analyses(
            paper_id, scope, provider, model, prompt_version, content_sha256,
            created_at, input_chars, summary, relevance_score, novelty_score,
            confidence, result_json, usage_json
        ) VALUES (1, 'abstract', 'ollama', 'old', 'v1', 'x', 'now', 1,
                  'legacy', 1, 1, 1, '{}', '{}');
        """
    )
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()

    with Database(path) as db:
        tables = {
            row[0]
            for row in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "analyses" in tables
        assert "analyses_v1_archive" in tables
        assert db._conn.execute("SELECT summary FROM analyses_v1_archive").fetchone()[0] == "legacy"
        columns = {
            row[1] for row in db._conn.execute("PRAGMA table_info(analyses)").fetchall()
        }
        assert "one_sentence_summary" in columns
        assert "relevance_score" not in columns
        assert db._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert ANALYSES_SCHEMA


def test_paper_record_reconstruction_and_author_tracking_state(tmp_path, author_candidate, paper_record):
    with Database(tmp_path / "records.sqlite") as db:
        db.add_author(author_candidate)
        paper_id, _ = db.upsert_paper(paper_record)
        reconstructed = db.get_paper_record(paper_id)
        assert reconstructed is not None
        assert reconstructed.title == paper_record.title
        assert [item.name for item in reconstructed.authors] == ["Ada Lovelace", "Alan Turing"]
        views = db.paper_authors(paper_id)
        assert views[0]["is_tracked"] == 1
        assert views[0]["tracked_enabled"] == 1
        assert views[1]["is_tracked"] is None
        assert db.get_paper_record(999999) is None


def test_analysis_task_lifecycle_and_interruption_recovery(tmp_path, paper_record):
    with Database(tmp_path / "tasks.sqlite") as db:
        paper_id, _ = db.upsert_paper(paper_record)
        ids = db.create_analysis_tasks(
            batch_id="batch-one",
            paper_id=paper_id,
            scope="abstract",
            profiles=[("one", "ollama", "m1"), ("two", "openai", "m2")],
        )
        assert len(ids) == 2
        db.start_analysis_task(ids[0])
        db.finish_analysis_task(ids[0], analysis_id=None)
        tasks = db.analysis_tasks_for_batch("batch-one")
        assert [row["status"] for row in tasks] == ["success", "queued"]
        assert db.fail_incomplete_analysis_tasks("restart") == 1
        tasks = db.analysis_tasks_for_paper(paper_id)
        assert {row["status"] for row in tasks} == {"success", "failed"}
        assert next(row for row in tasks if row["status"] == "failed")["error_text"] == "restart"


def test_v2_author_table_gains_non_destructive_tracking_column(tmp_path):
    import sqlite3
    from paperwatch.db import SCHEMA_VERSION

    path = tmp_path / "v2.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE authors (
            id INTEGER PRIMARY KEY, openalex_id TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
            orcid TEXT, institution TEXT, works_count INTEGER NOT NULL DEFAULT 0,
            cited_by_count INTEGER NOT NULL DEFAULT 0, enabled INTEGER NOT NULL DEFAULT 1,
            added_at TEXT NOT NULL, last_checked_at TEXT, metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        INSERT INTO authors(openalex_id, name, enabled, added_at)
        VALUES ('A42', 'Legacy Tracked Author', 1, '2026-01-01T00:00:00+00:00');
        """
    )
    conn.execute("PRAGMA user_version = 2")
    conn.commit()
    conn.close()

    with Database(path) as db:
        row = db.get_author("A42")
        assert row["tracked"] == 1
        assert db._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_author_tags_filter_suggestions_and_cleanup(tmp_path, author_candidate):
    from dataclasses import replace

    with Database(tmp_path / "tags.sqlite") as db:
        db.add_author(author_candidate)
        db.add_author(
            replace(
                author_candidate,
                openalex_id="A456",
                name="Grace Hopper",
                institution="United States Navy",
            )
        )
        assert db.add_author_tag("A123", " theory ")
        assert db.add_author_tag("A123", "ML")
        assert db.add_author_tag("A456", "ml")
        assert not db.add_author_tag("A999", "unknown")
        rows = db.list_authors_with_tags(tag="ML")
        assert [row["name"] for row in rows] == ["Ada Lovelace", "Grace Hopper"]
        rows = db.list_authors_with_tags(tags=("theory", "ML"))
        assert [row["name"] for row in rows] == ["Ada Lovelace", "Grace Hopper"]
        ada = db.list_authors_with_tags(tag="theory")[0]
        assert ada["tags"] == "ML, theory"
        db.set_object_notes("author", int(ada["id"]), "has notes")
        db.set_author_enabled("A456", False)
        assert [row["name"] for row in db.list_authors_with_tags(notes=True)] == [
            "Ada Lovelace"
        ]
        assert [row["name"] for row in db.list_authors_with_tags(enabled=False)] == [
            "Grace Hopper"
        ]
        assert [row["name"] for row in db.list_authors_with_tags(query="Engine")] == [
            "Ada Lovelace"
        ]
        assert [row["name"] for row in db.list_authors_with_tags(query="theory")] == [
            "Ada Lovelace"
        ]
        assert db.list_authors_with_tags(query="100%_missing") == []
        tags = {row["name"]: row["author_count"] for row in db.list_tags()}
        assert tags == {"ML": 2, "theory": 1}
        assert db.remove_author_tag("A123", "theory")
        assert not db.remove_author_tag("A123", "missing")
        assert "theory" not in {row["name"] for row in db.list_tags()}
        with pytest.raises(ValueError, match="empty"):
            db.add_author_tag("A123", " ")
        with pytest.raises(ValueError, match="40"):
            db.add_author_tag("A123", "x" * 41)


def test_search_papers_accepts_multiple_author_tag_toggles(
    tmp_path, author_candidate, paper_record
):
    from dataclasses import replace

    from paperwatch.models import PaperAuthor

    with Database(tmp_path / "paper-tags.sqlite") as db:
        db.add_author(author_candidate)
        db.add_author(
            replace(
                author_candidate,
                openalex_id="A456",
                name="Grace Hopper",
                institution="United States Navy",
            )
        )
        ada_id, _ = db.upsert_paper(paper_record)
        grace_id, _ = db.upsert_paper(
            replace(
                paper_record,
                openalex_id="W789",
                doi="10.1000/compiler",
                arxiv_id=None,
                title="Reliable Compiler Pipelines",
                authors=[PaperAuthor("A456", "Grace Hopper", "first", True)],
                topics=[],
            )
        )
        assert db.add_author_tag("A123", "theory")
        assert db.add_author_tag("A456", "systems")

        rows, total = db.search_papers(author_tags=("A123", "A456"), sort="title")
        assert total == 2
        assert {row["id"] for row in rows} == {ada_id, grace_id}

        rows, total = db.search_papers(author_tags=("https://openalex.org/A123",))
        assert total == 1
        assert rows[0]["id"] == ada_id

        # Existing custom author-group tags remain valid for old links and saved selections.
        rows, total = db.search_papers(
            author_tags=(" theory ", "", "systems"), sort="title"
        )
        assert total == 2
        assert {row["id"] for row in rows} == {ada_id, grace_id}
        rows, total = db.search_papers(author_tags=("THEORY",))
        assert total == 1
        assert rows[0]["id"] == ada_id
        assert db.search_papers(author_tags=("missing",))[1] == 0


def test_chunk_cache_and_saved_query_lifecycle(tmp_path, paper_record):
    with Database(tmp_path / "extra.sqlite") as db:
        paper_id, _ = db.upsert_paper(paper_record)
        assert db.get_analysis_chunk(
            paper_id=paper_id,
            provider="ollama",
            model="m",
            prompt_version="v",
            content_sha256="hash",
            chunk_index=0,
            context_window_tokens=1024,
        ) is None
        chunk_id = db.save_analysis_chunk(
            paper_id=paper_id,
            provider="ollama",
            model="m",
            prompt_version="v",
            content_sha256="hash",
            chunk_index=0,
            context_window_tokens=1024,
            summary={"section": "one"},
            usage={"tokens": 2},
        )
        assert chunk_id > 0
        assert db.save_analysis_chunk(
            paper_id=paper_id,
            provider="ollama",
            model="m",
            prompt_version="v",
            content_sha256="hash",
            chunk_index=0,
            context_window_tokens=1024,
            summary={"section": "two"},
            usage={},
        ) == chunk_id

        query_id = db.create_saved_query(
            name=" Compare ", source_type="papers", search_query="Minimal", prompt=" Compare methods "
        )
        assert db.get_saved_query(query_id)["name"] == "Compare"
        assert db.get_saved_query(query_id)["is_builtin"] == 0
        assert db.list_saved_queries()[0]["name"] == "Summarize"
        assert db.update_saved_query(
            query_id,
            name="Updated",
            source_type="summaries",
            search_query="Agents",
            prompt="Rank evidence",
        )
        run_ids = db.create_query_runs(
            batch_id="qbatch",
            query_id=query_id,
            profiles=[("default", "ollama", "m")],
            context_window_tokens=2048,
            selection={"query": "Minimal", "has_pdf": True},
        )
        db.start_query_run(run_ids[0])
        db.finish_query_run(
            run_ids[0],
            matched_count=1,
            input_chars=100,
            output={"title": "out", "summary": "s", "items": []},
        )
        row = db.list_query_runs(query_id)[0]
        assert row["status"] == "success"
        assert row["query_name"] == "Updated"
        assert __import__("json").loads(row["selection_json"])["has_pdf"] is True
        assert db.list_query_runs()[0]["id"] == run_ids[0]
        assert db.query_runs_for_batch("qbatch")[0]["output_json"]

        failed_ids = db.create_query_runs(
            batch_id="qbatch2",
            query_id=query_id,
            profiles=[("default", "ollama", "m")],
            context_window_tokens=2048,
        )
        assert db.fail_incomplete_query_runs("restart") == 1
        assert db.list_query_runs(query_id)[0]["status"] == "failed"
        assert db.delete_saved_query(query_id)
        assert db.get_saved_query(query_id) is None
        assert db.list_query_runs(query_id) == []
        assert not db.delete_saved_query(query_id)
        ignored_id = db.create_saved_query(
            name="x", source_type="bad", search_query="ignored", prompt="x"
        )
        ignored = db.get_saved_query(ignored_id)
        assert ignored["source_type"] == "papers"
        assert ignored["search_query"] == ""
        with pytest.raises(ValueError, match="required"):
            db.create_saved_query(name="", source_type="papers", search_query="", prompt="")
        assert not db.update_saved_query(
            999, name="x", source_type="papers", search_query="", prompt="x"
        )


def test_historical_paper_flag_is_persistent_and_local_pdf_counts(tmp_path, paper_record):
    path = tmp_path / "historical.sqlite"
    with Database(path) as db:
        paper_id, _ = db.upsert_paper(paper_record, auto_analyze=False)
        assert db.get_paper(paper_id)["auto_analyze"] == 0

        # A normal metadata refresh must not silently re-enable the historical backlog.
        db.upsert_paper(paper_record)
        assert db.get_paper(paper_id)["auto_analyze"] == 0

        db._conn.execute("UPDATE papers SET pdf_url=NULL WHERE id=?", (paper_id,))
        local = tmp_path / "paper.pdf"
        local.write_bytes(b"%PDF-1.4\n%%EOF")
        db.set_local_pdf(paper_id, str(local))
        assert db.dashboard_stats()["with_pdf"] == 1
        rows, total = db.search_papers(has_pdf=True)
        assert total == 1 and rows[0]["id"] == paper_id
        _, missing = db.search_papers(has_pdf=False)
        assert missing == 0


def test_version_six_database_gains_auto_analyze_column(tmp_path, paper_record):
    import sqlite3
    from paperwatch.db import SCHEMA_VERSION

    path = tmp_path / "v6.sqlite"
    with Database(path) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        assert db.get_paper(paper_id)["auto_analyze"] == 1

    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE papers DROP COLUMN auto_analyze")
    conn.execute("PRAGMA user_version = 6")
    conn.commit()
    conn.close()

    with Database(path) as db:
        row = db.get_paper(paper_id)
        assert row["auto_analyze"] == 1
        assert db._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert db._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_version_seven_database_gains_query_selection_column(tmp_path):
    import sqlite3
    from paperwatch.db import SCHEMA_VERSION

    path = tmp_path / "v7.sqlite"
    with Database(path) as db:
        query_id = db.create_saved_query(
            name="Legacy query", source_type="papers", search_query="", prompt="Compare"
        )
        run_id = db.create_query_runs(
            batch_id="legacy",
            query_id=query_id,
            profiles=[("default", "ollama", "m")],
            context_window_tokens=2048,
        )[0]

    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE query_runs DROP COLUMN selection_json")
    conn.execute("PRAGMA user_version = 7")
    conn.commit()
    conn.close()

    with Database(path) as db:
        row = db._conn.execute("SELECT * FROM query_runs WHERE id=?", (run_id,)).fetchone()
        assert row["selection_json"] == "{}"
        assert db._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert db._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_author_openalex_id_update_preserves_relationships(
    tmp_path, author_candidate, paper_record
):
    from dataclasses import replace

    from paperwatch.models import PaperAuthor

    with Database(tmp_path / "author-id.sqlite") as db:
        db.add_author(author_candidate)
        manual_id = db.add_manual_author("Manual Researcher", "Local Lab")
        manual = db.find_authors_by_name("Manual Researcher")[0]
        local_openalex_id = str(manual["openalex_id"])
        blog_id = db.add_author_blog(local_openalex_id, "https://example.test/manual")
        db.add_author_tag(local_openalex_id, "systems")
        manual_paper = replace(
            paper_record,
            openalex_id="W-MANUAL",
            doi="10.1000/manual",
            arxiv_id=None,
            title="Manual author paper",
            authors=[PaperAuthor(local_openalex_id, "Manual Researcher", "first", True)],
        )
        paper_id, _ = db.upsert_paper(manual_paper)
        run_id = db.start_run()
        db.record_error(
            run_id,
            stage="author",
            message="temporary",
            author_openalex_id=local_openalex_id,
        )

        assert db.update_author_openalex_id(local_openalex_id, "a999") is True
        assert db.get_author(local_openalex_id) is None
        updated = db.get_author("A999")
        assert updated is not None
        assert updated["id"] == manual_id
        assert updated["enabled"] == 1
        assert updated["last_checked_at"] is None
        assert db.author_blogs("A999")[0]["id"] == blog_id
        listed = next(
            row for row in db.list_authors_with_tags() if row["openalex_id"] == "A999"
        )
        assert listed["tags"] == "systems"
        paper_author = db._conn.execute(
            "SELECT * FROM paper_authors WHERE paper_id=?", (paper_id,)
        ).fetchone()
        assert paper_author["tracked_author_id"] == manual_id
        assert paper_author["openalex_author_id"] == "A999"
        error = db._conn.execute(
            "SELECT author_openalex_id FROM run_errors WHERE run_id=?", (run_id,)
        ).fetchone()
        assert error["author_openalex_id"] == "A999"

        assert db.update_author_openalex_id("A999", "A999") is True
        with pytest.raises(ValueError, match="already assigned"):
            db.update_author_openalex_id("A999", "A123")
        with pytest.raises(ValueError, match="must look like"):
            db.update_author_openalex_id("A999", "invalid")
        assert db.update_author_openalex_id("A404", "A888") is False

        db._conn.execute(
            "UPDATE authors SET metadata_json='not-json' WHERE openalex_id='A999'"
        )
        db._conn.commit()
        assert db.update_author_openalex_id("A999", "A998") is True
        db._conn.execute(
            "UPDATE authors SET metadata_json='[]' WHERE openalex_id='A998'"
        )
        db._conn.commit()
        assert db.update_author_openalex_id("A998", "A997") is True
