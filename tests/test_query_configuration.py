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
    assert 'name="model"' not in page.text
    assert 'name="paper_processing"' in page.text
    assert "Chunk + merge (default)" in page.text
    assert "Whole text (Codex only)" in page.text
    assert "PDF file (Codex only)" in page.text
    assert "Current default model:" in page.text

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
        assert "model_profile" not in row.keys()
        assert "model_profiles_json" not in row.keys()
        assert "context_window_tokens" not in row.keys()
        assert row["paper_source_scope"] == "auto"
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
        assert "model_profile" not in row.keys()
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


def test_default_query_installer_and_actions_column_deletion(config_path):
    client = _client(config_path)
    data_page = client.get("/data")
    assert data_page.status_code == 200
    assert 'action="/data/queries/defaults"' in data_page.text
    assert "Add default queries" in data_page.text

    config = load_config(config_path)
    with Database(config.paths.database) as db:
        initial_optional = [
            row for row in db.list_saved_queries() if not bool(row["is_builtin"])
        ]
        assert len(initial_optional) == 30

    installed = client.post("/data/queries/defaults", follow_redirects=False)
    assert installed.status_code == 303
    assert installed.headers["location"].startswith("/data?")
    assert "All+default+queries+are+already+present" in installed.headers["location"]

    with Database(config.paths.database) as db:
        rows = db.list_saved_queries()
        optional = [row for row in rows if not bool(row["is_builtin"])]
        assert len(optional) == 30
        assert sum(row["query_kind"] == "classification" for row in optional) == 10
        assert sum(row["regexp_mode"] == "print" for row in optional) == 10
        assert all(row["auto_run_papers"] == 0 for row in optional)
        assert all(row["auto_run_blogs"] == 0 for row in optional)
        assert all("model_profile" not in row.keys() for row in optional)
        assert all("model_profiles_json" not in row.keys() for row in optional)
        default_id = int(
            next(row for row in optional if row["name"] == "Evidence strength")["id"]
        )
        manual_id = db.create_saved_query(name="Manual query", prompt="Review this record")
        summary_id = int(next(row for row in rows if bool(row["is_builtin"]))["id"])

    repeated = client.post("/data/queries/defaults", follow_redirects=False)
    assert repeated.status_code == 303
    assert "All+default+queries+are+already+present" in repeated.headers["location"]

    page = client.get("/queries")
    assert page.status_code == 200
    assert 'class="saved-query-table"' in page.text
    assert 'class="saved-query-actions"' in page.text
    assert 'class="saved-query-action-grid"' in page.text
    assert 'saved-query-action-edit' in page.text
    assert 'saved-query-action-clear' in page.text
    assert 'saved-query-action-delete' in page.text
    assert 'href="/?query_id=' not in page.text
    assert 'href="/blogs?query_id=' not in page.text
    assert "Regexp filter" not in page.text
    assert page.text.count('data-filter-type="regexp"') == 10
    funding_at = page.text.index("Funding and grants")
    funding_row = page.text[page.text.rfind("<tr", 0, funding_at):page.text.index("</tr>", funding_at)]
    assert 'data-filter-type="regexp"' in funding_row
    assert ">Regexp</span>" in funding_row
    assert "Local regexp" in funding_row and "No model call" in funding_row
    assert "Clear tags" in funding_row

    css = client.get("/static/app.css")
    assert css.status_code == 200
    assert "table:not(.unified-content-table):not(.saved-query-table)" in css.text
    assert ".table-wrap>.saved-query-table{width:100%;min-width:100%;max-width:100%" in css.text
    history = client.get("/queries?mode=history")
    assert 'class="query-runs-table"' in history.text
    assert 'class="query-run-records-column"' in history.text
    assert ".query-run-records-column,.query-run-records-cell{width:7rem;max-width:7rem" in css.text

    for row in optional:
        assert f'action="/queries/{int(row["id"])}/delete"' in page.text
    assert f'action="/queries/{manual_id}/delete"' in page.text
    assert f'action="/queries/{summary_id}/delete"' not in page.text

    assert client.post(f"/queries/{default_id}/delete", follow_redirects=False).status_code == 303
    assert client.post(f"/queries/{manual_id}/delete", follow_redirects=False).status_code == 303
    protected = client.post(f"/queries/{summary_id}/delete", follow_redirects=False)
    assert protected.status_code == 303
    assert "cannot+be+deleted" in protected.headers["location"]
    with Database(config.paths.database) as db:
        assert db.get_saved_query(default_id) is None
        assert db.get_saved_query(manual_id) is None
        assert db.get_saved_query(summary_id) is not None


def test_collection_runner_places_override_and_options_after_query(
    config_path, paper_record
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.upsert_paper(paper_record)

    page = _client(config_path).get("/")
    assert page.status_code == 200
    runner = page.text.split('id="collection-runner-form"', 1)[1].split("</form>", 1)[0]
    assert 'class="collection-runner-row content-results-runner-row"' in runner
    assert 'class="collection-defaults-input"' in runner
    positions = [
        runner.index('name="query_id"'),
        runner.index("collection-defaults-toggle"),
        runner.index('name="model"'),
        runner.index('name="paper_processing"'),
        runner.index('name="context_window_tokens"'),
        runner.index("collection-run-button"),
    ]
    assert positions == sorted(positions)


def test_query_execution_validation_and_bulk_targets(config_path):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        query_from_legacy_models = db.create_saved_query(
            name="Legacy models ignored", prompt="P", model_profiles=("default", "secondary")
        )
        assert "model_profile" not in db.get_saved_query(query_from_legacy_models).keys()
        with pytest.raises(ValueError, match="Paper source"):
            db.create_saved_query(name="Bad", prompt="P", paper_source_scope="html")
        with pytest.raises(ValueError, match="PDF strategy"):
            db.create_saved_query(name="Bad", prompt="P", pdf_strategy="map")
        query_id = db.create_saved_query(
            name="Configured",
            prompt="P",
            model_profiles=("default", "default", ""),
        )
        assert "model_profile" not in db.get_saved_query(query_id).keys()
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
        assert "model_profile" not in columns
        assert "model_profiles_json" not in columns
        assert "context_window_tokens" not in columns
        assert db.get_query_run(run_id)["context_window_tokens"] == 7777
        assert db._conn.execute("PRAGMA user_version").fetchone()[0] == 32
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
    assert _configured_context_window(config, ("codex-review",), 2048) == 2048
    assert _configured_context_window(config, ("default", "codex-review"), 2048) == 2048


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
    assert kwargs["source_scope"] == "fulltext"
    assert kwargs["pdf_strategy"] == "direct"


def test_automatic_queries_use_current_config_default_model(
    app_config, paper_record, monkeypatch
):
    calls = []

    class DefaultLLM:
        provider = app_config.llm.provider
        model = app_config.llm.model

        def check_ready(self):
            calls.append(("ready", self.model))

        def close(self):
            pass

    class OpenAlex:
        def close(self):
            pass

    def collect(db, source_scope, **kwargs):
        calls.append(("collect", source_scope, kwargs["selection"]["pdf_strategy"]))
        return [{"paper_id": kwargs["selection"]["paper_ids"][0], "title": "Paper", "content": "text"}]

    def run_model(llm, **kwargs):
        calls.append(("run", llm.model, kwargs["context_window_tokens"], kwargs["pdf_strategy"]))
        return BatchQueryResult(title="Done", summary="Done", items=[]), {}, 4

    monkeypatch.setattr("paperwatch.query_jobs.collect_query_documents", collect)
    monkeypatch.setattr("paperwatch.query_jobs.run_query_for_model", run_model)

    with Database(app_config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        query_id = db.create_saved_query(
            name="Configured model",
            prompt="Review",
            auto_run_papers=True,
            # Legacy saved model fields are deliberately ignored.
            model_profiles=("secondary",),
            paper_source_scope="fulltext",
            pdf_strategy="direct",
        )
        app = PaperWatch(
            app_config, db=db, openalex=OpenAlex(), arxiv=None,
            llm=DefaultLLM(), pdf_client=httpx.Client(),
        )
        stats = RunStats(run_id=db.start_run(), new_paper_ids=[paper_id])
        app._run_automatic_queries(stats)
        run = next(row for row in db.list_query_runs() if int(row["query_id"]) == query_id)
        assert run["status"] == "success"
        assert run["profile"] == app_config.llm.profile
        assert run["model"] == app_config.llm.model
        assert stats.errors_count == 0

    assert any(call[0] == "collect" for call in calls)
    assert any(call[:2] == ("run", app_config.llm.model) for call in calls)

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
            "name": "Legacy model ignored",
            "prompt": "P",
            "model": "missing",
            "context_window_tokens": "2048",
        },
        follow_redirects=False,
    )
    assert unknown.status_code == 303 and "level=success" in unknown.headers["location"]
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


def test_automatic_query_default_model_and_collection_error_paths(
    app_config, paper_record, monkeypatch
):
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
        paper_id, _ = db.upsert_paper(paper_record)
        db.create_saved_query(name="Needs model", prompt="P", auto_run_papers=True)
        app = PaperWatch(
            no_llm_config, db=db, openalex=OpenAlex(), arxiv=None,
            llm=NoLLM(), pdf_client=httpx.Client(),
        )
        missing = RunStats(run_id=db.start_run(), new_paper_ids=[paper_id])
        app._run_automatic_queries(missing)
        assert missing.errors_count == 1

    # With a configured default, failures come from content/model execution, not
    # a stale model name stored on the query.
    with Database(app_config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        query_id = db.create_saved_query(
            name="Collection fails", prompt="P", auto_run_papers=True,
            model_profiles=("gone",),  # legacy value is ignored
        )
        app = PaperWatch(
            app_config, db=db, openalex=OpenAlex(), arxiv=None,
            llm=NoLLM(), pdf_client=httpx.Client(),
        )
        monkeypatch.setattr(
            "paperwatch.query_jobs.collect_query_documents",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("source failed")),
        )
        failed = RunStats(run_id=db.start_run(), new_paper_ids=[paper_id])
        app._run_automatic_queries(failed)
        assert failed.errors_count >= 1
        run = next(row for row in db.list_query_runs() if int(row["query_id"]) == query_id)
        assert run["status"] == "failed"
        assert run["model"] == app_config.llm.model

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


def test_regexp_tagger_can_be_created_from_query_form(config_path):
    client = _client(config_path)
    page = client.get("/queries/new")
    assert "Regexp tagger" in page.text
    assert "PDF file (Codex only)" in page.text
    assert "Whole text (Codex only)" in page.text
    assert "data-codex-only" in page.text

    response = client.post(
        "/queries",
        data={
            "name": "Compact architecture",
            "query_kind": "regexp",
            "content_scope": "papers",
            "regexp_pattern": r"(?i)compact\s+agent",
            "regexp_debug_text": "A compact agent architecture.",
            "paper_processing": "chunked",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "level=success" in response.headers["location"]

    config = load_config(config_path)
    with Database(config.paths.database) as db:
        row = next(
            item
            for item in db.list_saved_queries()
            if item["name"] == "Compact architecture"
        )
        assert row["query_kind"] == "regexp"
        assert "model_profile" not in row.keys()
        assert row["prompt"] == "Apply regexp tags locally."
        assert row["regexp_pattern"] == r"(?i)compact\s+agent"
        assert Database._decoded_label_values(row["label_values_json"]) == []
        assert row["regexp_debug_text"] == "A compact agent architecture."

    edit_page = client.get(f"/queries/{int(row['id'])}/edit")
    assert edit_page.status_code == 200
    assert 'name="regexp_debug_text"' in edit_page.text
    assert "A compact agent architecture." in edit_page.text

    updated = client.post(
        f"/queries/{int(row['id'])}/edit",
        data={
            "name": "Compact architecture",
            "query_kind": "regexp",
            "content_scope": "papers",
            "regexp_pattern": r"(?i)compact\s+agent",
            "regexp_debug_text": "Updated compact agent sample.",
            "paper_processing": "chunked",
        },
        follow_redirects=False,
    )
    assert updated.status_code == 303
    with Database(config.paths.database) as db:
        assert db.get_saved_query(int(row["id"]))["regexp_debug_text"] == (
            "Updated compact agent sample."
        )


def test_regexp_preprocessing_debug_text_is_saved_with_query(config_path):
    client = _client(config_path)
    response = client.post(
        "/queries",
        data={
            "name": "Retrieval excerpts",
            "prompt": "Summarize the matching excerpts.",
            "model": "default",
            "paper_source_scope": "auto",
            "pdf_strategy": "chunked",
            "regexp_pattern": r"(?i)retrieval|RAG",
            "regexp_radius": "50",
            "regexp_mode": "query",
            "regexp_debug_text": "This RAG system uses retrieval.",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303

    config = load_config(config_path)
    with Database(config.paths.database) as db:
        row = next(
            item
            for item in db.list_saved_queries()
            if item["name"] == "Retrieval excerpts"
        )
        assert row["regexp_pattern"] == r"(?i)retrieval|RAG"
        assert row["regexp_debug_text"] == "This RAG system uses retrieval."

    edit_page = client.get(f"/queries/{int(row['id'])}/edit")
    assert edit_page.status_code == 200
    assert 'name="regexp_debug_text"' in edit_page.text
    assert "This RAG system uses retrieval." in edit_page.text


def test_edit_preserves_regexp_debug_text_omitted_by_hidden_controls(config_path):
    client = _client(config_path)
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        query_id = db.create_saved_query(
            name="Regexp debug preservation",
            prompt="Summarize matches.",
            model_profiles=("default",),
            regexp_pattern=r"(?i)retrieval|RAG",
            regexp_radius=25,
            regexp_debug_text="Stored preprocessing sample.",
            query_kind="classification",
            label_class="Decision",
            label_values=[
                {"value": "Relevant", "pattern": r"(?i)yes|relevant"},
                {"value": "Irrelevant", "pattern": r"(?i)no|irrelevant"},
            ],
            label_debug_text="Stored classification sample.",
        )

    response = client.post(
        f"/queries/{query_id}/edit",
        data={
            "name": "Regexp debug preservation",
            "prompt": "Summarize matches.",
            "model": "default",
            "paper_source_scope": "auto",
            "pdf_strategy": "chunked",
            "query_kind": "classification",
            "content_scope": "both",
            "label_class": "Decision",
            "label_values": (
                "Relevant = (?i)yes|relevant\n"
                "Irrelevant = (?i)no|irrelevant"
            ),
            "regexp_pattern": r"(?i)retrieval|RAG",
            "regexp_radius": "25",
            "regexp_mode": "query",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303

    with Database(config.paths.database) as db:
        row = db.get_saved_query(query_id)
        assert row["regexp_debug_text"] == "Stored preprocessing sample."
        assert row["label_debug_text"] == "Stored classification sample."


def test_automatic_regexp_tagger_runs_without_llm(app_config, paper_record):
    class OpenAlex:
        def close(self):
            pass

    class NoLLM:
        provider = "none"
        model = "none"

        def close(self):
            pass

    with Database(app_config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        query_id = db.create_saved_query(
            name="Automatic compact tag",
            prompt="",
            auto_run_papers=True,
            query_kind="regexp",
            content_scope="papers",
            regexp_pattern=r"(?i)compact\s+agent",
            paper_source_scope="abstract",
        )
        app = PaperWatch(
            app_config,
            db=db,
            openalex=OpenAlex(),
            arxiv=None,
            llm=NoLLM(),
            pdf_client=httpx.Client(),
        )
        stats = RunStats(run_id=db.start_run(), new_paper_ids=[paper_id])
        app._run_automatic_queries(stats)

        run = next(row for row in db.list_query_runs() if row["query_id"] == query_id)
        assert run["status"] == "success"
        assert run["provider"] == "local"
        tags = db.classification_results_for_content("paper", paper_id)
        assert tags[0]["tag_name"].casefold() == "compact agent"
        assert stats.errors_count == 0
