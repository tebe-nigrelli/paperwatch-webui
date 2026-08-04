from __future__ import annotations

import json
import sqlite3
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient

from paperwatch.analysis_jobs import AnalysisBatchSnapshot
from paperwatch.config import load_config
from paperwatch.db import Database
from paperwatch.models import BatchQueryResult
from paperwatch.pipeline import PaperWatch, RunStats
from paperwatch.query_jobs import QueryBatchSnapshot
from paperwatch.web import (
    RunSnapshot,
    _configured_context_window,
    _paper_pdf_state,
    _paper_table_view,
    _parse_filter_values,
    _query_profile_names,
    _query_view,
    create_app,
)


class RunStub:
    def snapshot(self):
        return RunSnapshot()


class AnalysisStub:
    def snapshot(self):
        return AnalysisBatchSnapshot()


class QueryStub:
    def __init__(self):
        self.calls = []

    def snapshot(self):
        return QueryBatchSnapshot()

    def start(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "batch"


class AuthorClient:
    def close(self):
        pass


def _client(config_path, query=None):
    return TestClient(
        create_app(
            config_path,
            run_controller=RunStub(),
            analysis_controller=AnalysisStub(),
            query_controller=query or QueryStub(),
            author_client_factory=lambda config: AuthorClient(),
        )
    )


def test_query_and_paper_view_helper_edge_cases(config_path, tmp_path, monkeypatch):
    import paperwatch.web as web

    assert _query_profile_names("") == ()
    assert _query_profile_names("plain") == ("plain",)
    assert _query_profile_names('" encoded "') == ("encoded",)
    assert _query_profile_names(["one", "", "one", "two"]) == ("one", "two")
    assert _query_profile_names({"bad": "shape"}) == ()

    class Params:
        def get(self, name, default=None):
            return default

        def getlist(self, name):
            if name == "author_tag":
                return [f"author-{index}" for index in range(60)]
            if name == "tag":
                return [f"content-{index}" for index in range(60)]
            return []

    filters = _parse_filter_values(Params())
    assert len(filters.author_tags) == 50
    assert len(filters.content_tags) == 50

    config = load_config(config_path)
    view = _query_view(
        {
            "model_profile": "",
            "label_values_json": "{}",
            "regexp_radius": 0,
        },
        config,
    )
    assert view["model_profiles"] == ["default"]
    assert view["label_values"] == []

    class FakeDB:
        def __init__(self):
            self.hash_calls = []
            self.fingerprint_calls = []

        def set_paper_pdf_hash(self, *args, **kwargs):
            self.hash_calls.append((args, kwargs))
            return True

        def set_paper_pdf_fingerprint(self, *args, **kwargs):
            self.fingerprint_calls.append((args, kwargs))
            return True

    db = FakeDB()
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\ncontent\n%%EOF")
    stat = pdf.stat()

    legacy = _paper_pdf_state(
        db,
        {"id": 1, "local_pdf_path": str(pdf), "pdf_sha256": ""},
        verify_hash=False,
    )
    assert legacy["pdf_modified"] is False

    changed = _paper_pdf_state(
        db,
        {
            "id": 1,
            "local_pdf_path": str(pdf),
            "pdf_size": stat.st_size + 1,
            "pdf_mtime_ns": stat.st_mtime_ns,
            "pdf_sha256": "old",
        },
        verify_hash=False,
    )
    assert changed["pdf_modified"] is True

    monkeypatch.setattr(web, "pdf_sha256", lambda path: "fresh")
    refreshed = _paper_pdf_state(
        db,
        {"id": 2, "local_pdf_path": str(pdf), "pdf_sha256": ""},
    )
    assert refreshed["pdf_sha256"] == "fresh"
    assert db.hash_calls

    table = _paper_table_view(
        db,
        {
            "id": 3,
            "title": "Paper",
            "local_pdf_path": None,
            "tracked_authors_json": "{}",
            "author_count": 2,
        },
    )
    assert table["tracked_authors"] == [] and table["other_author_count"] == 2


def test_query_page_stores_one_model_and_uses_config_context(config_path):
    client = _client(config_path)
    page = client.get("/queries/new")
    assert page.status_code == 200
    assert "Context tokens" not in page.text
    assert 'name="context_window_tokens"' not in page.text
    model_field = page.text.split('name="model"', 1)[1].split("</select>", 1)[0]
    assert " multiple" not in model_field
    assert "Full paper PDF" in page.text
    assert 'value="default"' in page.text

    created = client.post(
        "/queries",
        data={
            "name": "Deep review",
            "prompt": "Review the evidence.",
            "model": "default",
            # Legacy clients may still submit this field; query definitions ignore it.
            "context_window_tokens": "4096",
            "paper_source_scope": "fulltext",
            "pdf_strategy": "chunked",
        },
        follow_redirects=False,
    )
    assert created.status_code == 303
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        row = next(item for item in db.list_saved_queries() if item["name"] == "Deep review")
        query_id = int(row["id"])
        assert row["model_profile"] == "default"
        assert "model_profiles_json" not in row.keys()
        assert "context_window_tokens" not in row.keys()
        assert row["paper_source_scope"] == "fulltext"
        assert row["pdf_strategy"] == "chunked"

    selected = client.post(
        "/queries/automatic",
        data={
            "query_id": str(query_id),
            "auto_run_papers": str(query_id),
            "auto_run_blogs": str(query_id),
        },
        follow_redirects=False,
    )
    assert selected.status_code == 303
    with Database(config.paths.database) as db:
        row = db.get_saved_query(query_id)
        assert row["auto_run_papers"] == 1
        assert row["auto_run_blogs"] == 1

    autosaved = client.post(
        "/queries/automatic",
        data={
            "query_id": str(query_id),
            "auto_run_blogs": str(query_id),
        },
        headers={"X-Paperwatch-Async": "1"},
    )
    assert autosaved.status_code == 200
    assert autosaved.json() == {
        "ok": True,
        "message": "Automatic query selections updated.",
    }
    with Database(config.paths.database) as db:
        row = db.get_saved_query(query_id)
        assert row["auto_run_papers"] == 0
        assert row["auto_run_blogs"] == 1

    edited = client.post(
        f"/queries/{query_id}/edit",
        data={
            "name": "Deep review",
            "prompt": "Review the evidence and caveats.",
            "model": "default",
            # Legacy clients may submit this field; it is not persisted.
            "context_window_tokens": "8192",
            "paper_source_scope": "abstract",
            "pdf_strategy": "direct",
        },
        follow_redirects=False,
    )
    assert edited.status_code == 303
    with Database(config.paths.database) as db:
        row = db.get_saved_query(query_id)
        assert "context_window_tokens" not in row.keys()
        assert row["model_profile"] == "default"
        assert row["pdf_strategy"] == "auto"
        assert row["auto_run_papers"] == 0
        assert row["auto_run_blogs"] == 1

    table = client.get("/queries")
    assert "New papers" in table.text
    assert "New blog entries" in table.text
    assert "Deep review" in table.text
    assert 'href="/queries/new"' in table.text
    assert "Run backward" not in table.text
    assert "Save selections" not in table.text
    assert "Changes save automatically" in table.text


def test_query_execution_validation_and_bulk_targets(config_path):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        with pytest.raises(ValueError, match="only one model"):
            db.create_saved_query(
                name="Bad", prompt="P", model_profiles=("default", "secondary")
            )
        with pytest.raises(ValueError, match="Paper source"):
            db.create_saved_query(name="Bad", prompt="P", paper_source_scope="html")
        with pytest.raises(ValueError, match="PDF strategy"):
            db.create_saved_query(name="Bad", prompt="P", pdf_strategy="map")
        query_id = db.create_saved_query(
            name="Configured",
            prompt="P",
            model_profiles=("default", "default", ""),
        )
        assert db.get_saved_query(query_id)["model_profile"] == "default"
        db.update_automatic_queries(
            [query_id], paper_query_ids=[query_id], blog_query_ids=[]
        )
        assert db.get_saved_query(query_id)["auto_run_papers"] == 1
        db.update_automatic_queries(
            [query_id], paper_query_ids=[], blog_query_ids=[query_id]
        )
        row = db.get_saved_query(query_id)
        assert row["auto_run_papers"] == 0 and row["auto_run_blogs"] == 1



def test_schema_25_migrates_saved_query_model_and_removes_context(tmp_path):
    path = tmp_path / "legacy-v24.db"
    with Database(path) as db:
        query_id = db.create_saved_query(
            name="Legacy query",
            prompt="Review",
            model_profiles=("secondary",),
        )
        run_id = db.create_query_runs(
            batch_id="legacy-batch",
            query_id=query_id,
            profiles=[("secondary", "ollama", "legacy-model")],
            context_window_tokens=7777,
        )[0]

    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = OFF")
    row = conn.execute("SELECT * FROM saved_queries WHERE id=?", (query_id,)).fetchone()
    columns = [item[1] for item in conn.execute("PRAGMA table_info(saved_queries)")]
    values = dict(zip(columns, row, strict=True))
    conn.execute("DROP TABLE saved_queries")
    conn.execute(
        """
        CREATE TABLE saved_queries (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            source_type TEXT NOT NULL,
            search_query TEXT NOT NULL DEFAULT '',
            prompt TEXT NOT NULL,
            is_builtin INTEGER NOT NULL DEFAULT 0,
            auto_run_papers INTEGER NOT NULL DEFAULT 0,
            auto_run_blogs INTEGER NOT NULL DEFAULT 0,
            model_profiles_json TEXT NOT NULL DEFAULT '[]',
            context_window_tokens INTEGER NOT NULL DEFAULT 16384,
            paper_source_scope TEXT NOT NULL DEFAULT 'auto',
            pdf_strategy TEXT NOT NULL DEFAULT 'chunked',
            regexp_pattern TEXT NOT NULL DEFAULT '',
            regexp_radius INTEGER NOT NULL DEFAULT 0,
            regexp_mode TEXT NOT NULL DEFAULT 'query',
            content_scope TEXT NOT NULL DEFAULT 'both',
            query_kind TEXT NOT NULL DEFAULT 'summary',
            label_class TEXT NOT NULL DEFAULT '',
            label_values_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_run_at TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO saved_queries(
            id, name, source_type, search_query, prompt, is_builtin,
            auto_run_papers, auto_run_blogs, model_profiles_json,
            context_window_tokens, paper_source_scope, pdf_strategy,
            regexp_pattern, regexp_radius, regexp_mode, content_scope,
            query_kind, label_class, label_values_json, created_at,
            updated_at, last_run_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            values["id"], values["name"], values["source_type"],
            values["search_query"], values["prompt"], values["is_builtin"],
            values["auto_run_papers"], values["auto_run_blogs"],
            json.dumps(["secondary", "default"]), 9999,
            values["paper_source_scope"], values["pdf_strategy"],
            values["regexp_pattern"], values["regexp_radius"],
            values["regexp_mode"], values["content_scope"],
            values["query_kind"], values["label_class"],
            values["label_values_json"], values["created_at"],
            values["updated_at"], values["last_run_at"],
        ),
    )
    conn.execute("PRAGMA user_version = 24")
    conn.commit()
    conn.close()

    with Database(path) as db:
        columns = [row["name"] for row in db._conn.execute("PRAGMA table_info(saved_queries)")]
        assert "model_profile" in columns
        assert "model_profiles_json" not in columns
        assert "context_window_tokens" not in columns
        assert db.get_saved_query(query_id)["model_profile"] == "secondary"
        assert db.get_query_run(run_id)["context_window_tokens"] == 7777
        assert db._conn.execute("PRAGMA user_version").fetchone()[0] == 25
        assert db._conn.execute("PRAGMA foreign_key_check").fetchall() == []

def test_context_window_override_is_local_only(app_config):
    codex = replace(
        app_config.llm,
        profile="codex-review",
        provider="codex",
        model="gpt-5.3-codex",
        context_window_tokens=400000,
    )
    config = replace(app_config, llm_profiles=(codex,))

    assert _configured_context_window(config, ("default",), 2048) == 2048
    assert _configured_context_window(config, ("codex-review",), 2048) == 400000
    with pytest.raises(ValueError, match="only selected model"):
        _configured_context_window(
            config,
            ("default", "codex-review"),
            2048,
        )


def test_backward_run_uses_saved_model_and_config_context(config_path, paper_record):
    controller = QueryStub()
    client = _client(config_path, controller)
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        query_id = db.create_saved_query(
            name="Stored settings",
            prompt="P",
            model_profiles=("default",),
            context_window_tokens=7777,
            paper_source_scope="abstract",
            pdf_strategy="direct",
        )
        # Give the backward paper run one eligible record.
        paper_record.abstract = "Abstract"
        db.upsert_paper(paper_record)
    response = client.post(
        f"/queries/{query_id}/run-existing/papers", follow_redirects=False
    )
    assert response.status_code == 303
    args, kwargs = controller.calls[-1]
    assert args[:3] == (
        query_id,
        ("default",),
        config.llm.context_window_tokens,
    )
    assert kwargs["source_scope"] == "abstract"
    assert kwargs["pdf_strategy"] == "auto"


def test_automatic_queries_use_saved_model_config_context(
    app_config, paper_record, analysis_result, monkeypatch
):
    secondary = replace(
        app_config.llm,
        profile="secondary",
        model="second-model",
        context_window_tokens=24000,
    )
    config = replace(app_config, llm_profiles=(secondary,))
    calls = []

    class SecondaryLLM:
        provider = "fake"
        model = "second-model"
        closed = False

        def check_ready(self):
            calls.append(("ready", self.model))

        def close(self):
            self.closed = True
            calls.append(("closed", self.model))

    class DefaultLLM:
        provider = "fake"
        model = "fake-model"

        def close(self):
            pass

    class OpenAlex:
        def close(self):
            pass

    def collect(db, source_scope, **kwargs):
        calls.append(("collect", source_scope, kwargs["selection"]["pdf_strategy"]))
        return [{"paper_id": 1, "title": "Paper", "content": "text"}]

    def run_model(llm, **kwargs):
        calls.append(
            (
                "run",
                llm.model,
                kwargs["context_window_tokens"],
                kwargs["pdf_strategy"],
            )
        )
        return BatchQueryResult(title="Done", summary="Done", items=[]), {}, 4

    monkeypatch.setattr("paperwatch.pipeline.create_llm", lambda profile: SecondaryLLM())
    monkeypatch.setattr("paperwatch.query_jobs.collect_query_documents", collect)
    monkeypatch.setattr("paperwatch.query_jobs.run_query_for_model", run_model)

    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        db.create_saved_query(
            name="Configured model",
            prompt="Review",
            auto_run_papers=True,
            model_profiles=("secondary",),
            context_window_tokens=6000,
            paper_source_scope="fulltext",
            pdf_strategy="direct",
        )
        db.create_saved_query(
            name="Missing model",
            prompt="Review",
            auto_run_blogs=True,
            model_profiles=("gone",),
        )
        app = PaperWatch(
            config,
            db=db,
            openalex=OpenAlex(),
            arxiv=None,
            llm=DefaultLLM(),
            pdf_client=httpx.Client(),
        )
        stats = RunStats(
            run_id=db.start_run(), new_paper_ids=[paper_id], new_blog_entry_ids=[9]
        )
        app._run_automatic_queries(stats)
        rows = [row for row in db.list_query_runs() if row["query_id"] != 1]
        assert {row["model"] for row in rows} == {"second-model"}
        assert {row["context_window_tokens"] for row in rows} == {
            secondary.context_window_tokens
        }
        assert all(row["status"] == "success" for row in rows)
        assert stats.errors_count == 1

        monkeypatch.setattr(
            "paperwatch.query_jobs.run_query_for_model",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("model failed")),
        )
        failed = RunStats(run_id=db.start_run(), new_paper_ids=[paper_id])
        app._run_automatic_queries(failed)
        assert failed.errors_count >= 1
        assert db.list_query_runs()[0]["status"] == "failed"
    assert ("collect", "fulltext", "direct") in calls
    assert (
        "run",
        "second-model",
        secondary.context_window_tokens,
        "direct",
    ) in calls
    assert ("ready", "second-model") in calls
    assert ("closed", "second-model") in calls


def test_download_only_historical_run_skips_blogs(app_config, author_candidate, monkeypatch):
    config = replace(app_config, llm=replace(app_config.llm, provider="none"))

    class OpenAlex:
        def recent_works(self, author_id, *, since, max_results):
            return []

        def close(self):
            pass

    class NoLLM:
        provider = "none"
        model = "none"

        def close(self):
            pass

    monkeypatch.setattr(
        "paperwatch.pipeline.fetch_blog",
        lambda *args, **kwargs: pytest.fail("historical backfill fetched a blog"),
    )
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        db.add_author_blog(author_candidate.openalex_id, "https://example.test/blog")
        app = PaperWatch(
            config,
            db=db,
            openalex=OpenAlex(),
            arxiv=None,
            llm=NoLLM(),
            pdf_client=httpx.Client(),
        )
        stats = app.run(download_only=True, historical=True)
        assert stats.status == "success"
        assert stats.blogs_checked == 0


def test_query_form_validation_and_missing_routes(config_path):
    client = _client(config_path)
    unknown = client.post(
        "/queries",
        data={
            "name": "Bad model",
            "prompt": "P",
            "model": "missing",
            "context_window_tokens": "2048",
        },
        follow_redirects=False,
    )
    assert unknown.status_code == 303 and "level=error" in unknown.headers["location"]
    legacy_context = client.post(
        "/queries",
        data={
            "name": "Tiny",
            "prompt": "P",
            "model": "default",
            "context_window_tokens": "10",
        },
        follow_redirects=False,
    )
    assert legacy_context.status_code == 303
    assert "level=success" in legacy_context.headers["location"]
    assert client.get("/queries/999999/edit").status_code == 404
    missing_edit = client.post(
        "/queries/999999/edit",
        data={
            "name": "Missing",
            "prompt": "P",
            "model": "default",
            "paper_source_scope": "auto",
            "pdf_strategy": "chunked",
        },
    )
    assert missing_edit.status_code == 404
    cleared = client.post(
        "/queries/automatic",
        data={"query_id": ["bad", "999999"], "auto_run_papers": "bad"},
        follow_redirects=False,
    )
    assert cleared.status_code == 303


def test_automatic_query_profile_and_collection_error_paths(
    app_config, paper_record, monkeypatch
):
    assert PaperWatch._stored_query_profiles({"model_profiles_json": "{"}) == ()
    assert PaperWatch._stored_query_profiles({"model_profiles_json": "{}"}) == ()

    class OpenAlex:
        def close(self):
            pass

    class NoLLM:
        provider = "none"
        model = "none"

        def close(self):
            pass

    no_llm_config = replace(app_config, llm=replace(app_config.llm, provider="none"))
    with Database(no_llm_config.paths.database) as db:
        app = PaperWatch(
            no_llm_config,
            db=db,
            openalex=OpenAlex(),
            arxiv=None,
            llm=NoLLM(),
            pdf_client=httpx.Client(),
        )
        app._run_automatic_queries(RunStats(run_id=db.start_run(), new_paper_ids=[1]))

    # Re-open with an available model and exercise the no-valid-model and source failure paths.
    with Database(app_config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        db.create_saved_query(
            name="Unavailable only",
            prompt="P",
            auto_run_blogs=True,
            model_profiles=("gone",),
        )
        app = PaperWatch(
            app_config,
            db=db,
            openalex=OpenAlex(),
            arxiv=None,
            llm=NoLLM(),
            pdf_client=httpx.Client(),
        )
        missing = RunStats(run_id=db.start_run(), new_blog_entry_ids=[1])
        app._run_automatic_queries(missing)
        assert missing.errors_count == 1

        db.create_saved_query(
            name="Collection fails",
            prompt="P",
            auto_run_papers=True,
            model_profiles=("default",),
        )
        monkeypatch.setattr(
            "paperwatch.query_jobs.collect_query_documents",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("source failed")),
        )
        failed = RunStats(run_id=db.start_run(), new_paper_ids=[paper_id])
        app._run_automatic_queries(failed)
        assert failed.errors_count >= 1
        assert any(row["status"] == "failed" for row in db.list_query_runs())


def test_automatic_query_cancellation_marks_run(
    app_config, paper_record, monkeypatch
):
    from paperwatch.job_runtime import JobCancelled

    class OpenAlex:
        def close(self):
            pass

    class LLM:
        provider = "fake"
        model = "fake-model"

        def close(self):
            pass

    monkeypatch.setattr(
        "paperwatch.query_jobs.collect_query_documents",
        lambda *args, **kwargs: [{"paper_id": 1, "title": "Paper", "content": "text"}],
    )
    monkeypatch.setattr(
        "paperwatch.query_jobs.run_query_for_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(JobCancelled("cancelled")),
    )
    with Database(app_config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        db.create_saved_query(
            name="Cancel",
            prompt="P",
            auto_run_papers=True,
            model_profiles=("default",),
        )
        app = PaperWatch(
            app_config,
            db=db,
            openalex=OpenAlex(),
            arxiv=None,
            llm=LLM(),
            pdf_client=httpx.Client(),
        )
        stats = RunStats(run_id=db.start_run(), new_paper_ids=[paper_id])
        with pytest.raises(JobCancelled):
            app._run_automatic_queries(stats)
        assert db.list_query_runs()[0]["error_text"] == "Cancelled by user"
