from __future__ import annotations

import json
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
from paperwatch.web import RunSnapshot, create_app


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


def test_query_page_stores_execution_settings_and_automatic_columns(config_path):
    client = _client(config_path)
    page = client.get("/queries/new")
    assert page.status_code == 200
    assert "Context tokens" in page.text
    assert "Full paper PDF" in page.text
    assert 'value="default"' in page.text

    created = client.post(
        "/queries",
        data={
            "name": "Deep review",
            "prompt": "Review the evidence.",
            "model": "default",
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
        assert json.loads(row["model_profiles_json"]) == ["default"]
        assert row["context_window_tokens"] == 4096
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
            "context_window_tokens": "8192",
            "paper_source_scope": "abstract",
            "pdf_strategy": "direct",
        },
        follow_redirects=False,
    )
    assert edited.status_code == 303
    with Database(config.paths.database) as db:
        row = db.get_saved_query(query_id)
        assert row["context_window_tokens"] == 8192
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
        with pytest.raises(ValueError, match="Context window"):
            db.create_saved_query(name="Bad", prompt="P", context_window_tokens=10)
        with pytest.raises(ValueError, match="Paper source"):
            db.create_saved_query(name="Bad", prompt="P", paper_source_scope="html")
        with pytest.raises(ValueError, match="PDF strategy"):
            db.create_saved_query(name="Bad", prompt="P", pdf_strategy="map")
        query_id = db.create_saved_query(
            name="Configured",
            prompt="P",
            model_profiles=("default", "default", ""),
        )
        assert json.loads(db.get_saved_query(query_id)["model_profiles_json"]) == ["default"]
        db.update_automatic_queries(
            [query_id], paper_query_ids=[query_id], blog_query_ids=[]
        )
        assert db.get_saved_query(query_id)["auto_run_papers"] == 1
        db.update_automatic_queries(
            [query_id], paper_query_ids=[], blog_query_ids=[query_id]
        )
        row = db.get_saved_query(query_id)
        assert row["auto_run_papers"] == 0 and row["auto_run_blogs"] == 1


def test_backward_run_uses_saved_models_and_context(config_path, paper_record):
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
    assert args[:3] == (query_id, ("default",), 7777)
    assert kwargs["source_scope"] == "abstract"
    assert kwargs["pdf_strategy"] == "auto"


def test_automatic_queries_use_each_saved_model_and_context(
    app_config, paper_record, analysis_result, monkeypatch
):
    secondary = replace(app_config.llm, profile="secondary", model="second-model")
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
            name="Two models",
            prompt="Review",
            auto_run_papers=True,
            model_profiles=("default", "secondary"),
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
        assert {row["model"] for row in rows} == {"test-model", "second-model"}
        assert {row["context_window_tokens"] for row in rows} == {6000}
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
    assert ("run", "fake-model", 6000, "direct") in calls
    assert ("run", "second-model", 6000, "direct") in calls
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
    tiny = client.post(
        "/queries",
        data={
            "name": "Tiny",
            "prompt": "P",
            "model": "default",
            "context_window_tokens": "10",
        },
        follow_redirects=False,
    )
    assert tiny.status_code == 303 and "level=error" in tiny.headers["location"]
    assert client.get("/queries/999999/edit").status_code == 404
    missing_edit = client.post(
        "/queries/999999/edit",
        data={
            "name": "Missing",
            "prompt": "P",
            "model": "default",
            "context_window_tokens": "2048",
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
