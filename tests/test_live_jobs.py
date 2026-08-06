from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict

import httpx
import pytest
from fastapi.testclient import TestClient

from paperwatch.analysis_jobs import AnalysisBatchSnapshot, AnalysisController, run_analysis_batch
from paperwatch.config import load_config
from paperwatch.db import Database
from paperwatch.job_runtime import JobCancelled, check_cancelled, publish
from paperwatch.llm import OllamaClient
from paperwatch.models import BATCH_QUERY_JSON_SCHEMA, LLMResponse
from paperwatch.query_jobs import QueryBatchSnapshot, QueryController, collect_query_documents, run_query_batch
from paperwatch.web import RunSnapshot, create_app


class RuntimeAnalysisLLM:
    provider = "ollama"

    def __init__(self, result, *, model="runtime-model"):
        self.model = model
        self.result = result
        self.progress = None
        self.cancel_event = None
        self.closed = False

    def set_runtime(self, progress, cancel_event):
        self.progress = progress
        self.cancel_event = cancel_event

    def analyze(self, prompt):
        check_cancelled(self.cancel_event)
        if self.progress:
            self.progress("output", '{"one_sentence_summary":"live')
            self.progress("output", ' output"}')
        check_cancelled(self.cancel_event)
        return LLMResponse(result=self.result, usage={"eval_count": 2})

    def close(self):
        self.closed = True


class RuntimeQueryLLM:
    provider = "ollama"
    model = "runtime-query"

    def __init__(self):
        self.progress = None
        self.cancel_event = None
        self.closed = False

    def set_runtime(self, progress, cancel_event):
        self.progress = progress
        self.cancel_event = cancel_event

    def generate_json(self, prompt, *, schema, schema_name):
        from paperwatch.llm import StructuredJSONResponse

        check_cancelled(self.cancel_event)
        payload = {
            "title": "Single-paper prompt",
            "summary": "The saved prompt completed.",
            "items": [{"paper_id": 1, "title": "Paper", "output": "Result"}],
        }
        if self.progress:
            self.progress("output", json.dumps(payload))
        return StructuredJSONResponse(data=payload, usage={}, raw=payload)

    def close(self):
        self.closed = True


def test_ollama_streams_structured_output_and_can_cancel(analysis_result):
    lines = [
        {"message": {"content": analysis_result.model_dump_json()[:80]}, "done": False},
        {
            "message": {"content": analysis_result.model_dump_json()[80:]},
            "done": True,
            "eval_count": 7,
        },
    ]
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        body = "".join(json.dumps(item) + "\n" for item in lines)
        return httpx.Response(200, text=body)

    output = []
    llm = OllamaClient(
        model="qwen",
        base_url="http://ollama.test",
        timeout_seconds=1,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    llm.set_runtime(lambda kind, value: output.append((kind, value)), threading.Event())
    response = llm.analyze("paper")
    assert seen["stream"] is True
    assert response.usage["eval_count"] == 7
    assert "".join(value for kind, value in output if kind == "output") == analysis_result.model_dump_json()

    event = threading.Event()

    def cancel_after_first(kind, value):
        output.append((kind, value))
        event.set()

    llm.set_runtime(cancel_after_first, event)
    with pytest.raises(JobCancelled, match="Cancelled"):
        llm.generate_json("paper", schema=BATCH_QUERY_JSON_SCHEMA, schema_name="x")


def test_analysis_batch_reports_live_output_and_cancelled_tasks(
    config_path, paper_record, analysis_result, monkeypatch
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        db.create_analysis_tasks(
            batch_id="live",
            paper_id=paper_id,
            scope="abstract",
            profiles=[("default", "ollama", "test-model")],
        )
    client = RuntimeAnalysisLLM(analysis_result)
    monkeypatch.setattr("paperwatch.analysis_jobs.create_llm", lambda profile: client)
    events = []
    active = []
    result = run_analysis_batch(
        config_path,
        "live",
        paper_id,
        "abstract",
        ("default",),
        progress=lambda kind, value: events.append((kind, value)),
        cancel_event=threading.Event(),
        active_client=lambda value: active.append(value),
    )
    assert result == {"completed": 1, "failed": 0}
    assert any(kind == "output" for kind, _ in events)
    assert active[0] is client and active[-1] is None
    assert client.closed

    with Database(config.paths.database) as db:
        db.create_analysis_tasks(
            batch_id="cancelled",
            paper_id=paper_id,
            scope="abstract",
            profiles=[("default", "ollama", "test-model")],
        )
    cancelled = threading.Event()
    cancelled.set()
    assert run_analysis_batch(
        config_path,
        "cancelled",
        paper_id,
        "abstract",
        ("default",),
        cancel_event=cancelled,
    )["cancelled"] is True
    with Database(config.paths.database) as db:
        task = db.analysis_tasks_for_batch("cancelled")[0]
        assert task["status"] == "failed"
        assert task["error_text"] == "Cancelled by user"


def test_analysis_controller_progress_and_cancel(config_path, paper_record):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
    entered = threading.Event()

    class Active:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    active = Active()

    def runner(path, batch, paper, scope, profiles, strategy, context, progress, cancel, active_client):
        progress("stage", "Generating")
        progress("reset_output", "")
        progress("output", "partial")
        active_client(active)
        entered.set()
        cancel.wait(2)
        active_client(None)
        return {"completed": 0, "failed": 0, "cancelled": True}

    controller = AnalysisController(config_path, runner=runner)
    assert controller.start(paper_id, "abstract", ("default",))
    assert entered.wait(1)
    assert controller.snapshot().live_output == "partial"
    assert controller.cancel()
    controller.wait()
    snapshot = controller.snapshot()
    assert snapshot.cancelled and snapshot.stage == "Cancelled"
    assert active.closed
    assert controller.cancel() is False


def test_query_single_paper_live_output_and_controller_cancel(
    config_path, paper_record, monkeypatch
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        query_id = db.create_saved_query(
            name="Critique", source_type="papers", search_query="", prompt="Critique this paper"
        )
        db.create_query_runs(
            batch_id="paper-prompt",
            query_id=query_id,
            profiles=[("default", "ollama", "test-model")],
            context_window_tokens=2048,
            selection={"paper_ids": [paper_id]},
        )
        documents = collect_query_documents(
            db, "papers", selection={"paper_ids": [paper_id, "bad", 999999]}
        )
        assert [item["paper_id"] for item in documents] == [paper_id]

    llm = RuntimeQueryLLM()
    monkeypatch.setattr("paperwatch.query_jobs.create_llm", lambda profile: llm)
    events = []
    result = run_query_batch(
        config_path,
        "paper-prompt",
        query_id,
        ("default",),
        2048,
        progress=lambda kind, value: events.append((kind, value)),
        cancel_event=threading.Event(),
    )
    assert result == {"completed": 1, "failed": 0}
    assert any(kind == "output" for kind, _ in events)
    with Database(config.paths.database) as db:
        rows = db.query_runs_for_paper(paper_id)
        assert len(rows) == 1 and rows[0]["status"] == "success"

    entered = threading.Event()

    def runner(path, batch, query, profiles, context, progress, cancel, active_client):
        progress("stage", "Running")
        entered.set()
        cancel.wait(2)
        return {"completed": 0, "failed": 0, "cancelled": True}

    controller = QueryController(config_path, runner=runner)
    assert controller.start(
        query_id, ("default",), 2048, selection={"paper_ids": [paper_id]}
    )
    assert entered.wait(1)
    assert controller.snapshot().paper_id == paper_id
    assert controller.cancel()
    controller.wait()
    assert controller.snapshot().cancelled
    assert controller.cancel() is False


def test_database_hides_inconclusive_tasks_and_prompt_runs(
    config_path, paper_record
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        query_id = db.create_saved_query(
            name="Q", source_type="papers", search_query="", prompt="P"
        )
        task_ids = db.create_analysis_tasks(
            batch_id="tasks",
            paper_id=paper_id,
            scope="abstract",
            profiles=[("ok", "ollama", "m1"), ("bad", "ollama", "m2")],
        )
        db.finish_analysis_task(task_ids[0], analysis_id=None)
        db.finish_analysis_task(task_ids[1], error="bad")
        assert [row["profile"] for row in db.visible_analysis_tasks_for_paper(paper_id)] == ["ok"]

        run_ids = db.create_query_runs(
            batch_id="runs",
            query_id=query_id,
            profiles=[("ok", "ollama", "m1"), ("bad", "ollama", "m2")],
            context_window_tokens=2048,
            selection={"paper_ids": [paper_id]},
        )
        db.finish_query_run(
            run_ids[0], matched_count=1, input_chars=1,
            output={"title": "T", "summary": "S", "items": []},
        )
        db.finish_query_run(run_ids[1], matched_count=1, input_chars=1, error="bad")
        assert [row["profile"] for row in db.query_runs_for_paper(paper_id)] == ["ok"]
        assert len(db.query_runs_for_paper(paper_id, successful_only=False)) == 2


class RunStub:
    def snapshot(self):
        return RunSnapshot()

    def start(self, mode="run"):
        return True


class AnalysisWebStub:
    def __init__(self):
        self.state = AnalysisBatchSnapshot()
        self.cancelled = False

    def snapshot(self):
        return AnalysisBatchSnapshot(**asdict(self.state))

    def start(self, paper_id, scope, profiles, analysis_strategy="direct", context_window_tokens=None):
        self.state = AnalysisBatchSnapshot(
            running=True, batch_id="analysis-web", paper_id=paper_id, scope=scope,
            profiles=profiles, stage="Queued", message="Starting",
        )
        return "analysis-web"

    def cancel(self):
        self.cancelled = True
        return True


class QueryWebStub:
    def __init__(self):
        self.state = QueryBatchSnapshot()
        self.calls = []
        self.cancelled = False

    def snapshot(self):
        return QueryBatchSnapshot(**asdict(self.state))

    def start(
        self, query_id, profiles, context, *, selection=None,
        source_scope="abstract", pdf_strategy="auto",
    ):
        self.calls.append((query_id, profiles, context, selection, source_scope, pdf_strategy))
        self.state = QueryBatchSnapshot(
            running=True, batch_id="query-web", query_id=query_id,
            paper_id=selection["paper_ids"][0], profiles=profiles, stage="Queued",
        )
        return "query-web"

    def cancel(self):
        self.cancelled = True
        return True


class AuthorStub:
    def close(self):
        pass


def test_paper_async_jobs_saved_prompt_and_no_reload_ui(
    config_path, author_candidate, paper_record
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        paper_id, _ = db.upsert_paper(paper_record)
        query_id = db.create_saved_query(
            name="Saved critique", source_type="papers", search_query="", prompt="Critique",
            model_profiles=("default",), paper_source_scope="fulltext",
            pdf_strategy="direct",
        )
        failed = db.create_analysis_tasks(
            batch_id="old", paper_id=paper_id, scope="abstract",
            profiles=[("default", "ollama", "test-model")],
        )[0]
        db.finish_analysis_task(failed, error="old failure")
        run_id = db.create_query_runs(
            batch_id="old-prompt", query_id=query_id,
            profiles=[("default", "ollama", "test-model")],
            context_window_tokens=2048, selection={"paper_ids": [paper_id]},
        )[0]
        db.finish_query_run(
            run_id, matched_count=1, input_chars=10,
            output={
                "title": "Saved result", "summary": "Prompt output",
                "items": [{"paper_id": paper_id, "title": paper_record.title, "output": "Critical result"}],
            },
        )

    analysis = AnalysisWebStub()
    query = QueryWebStub()
    client = TestClient(
        create_app(
            config_path,
            run_controller=RunStub(),
            analysis_controller=analysis,
            query_controller=query,
            author_client_factory=lambda config: AuthorStub(),
        )
    )
    page = client.get(f"/papers/{paper_id}")
    assert 'data-paper-job="analysis"' not in page.text
    assert 'data-paper-job="query"' not in page.text
    assert 'data-query-defaults-form' not in page.text
    assert "Saved critique" in page.text
    assert "Saved result" in page.text and "Critical result" in page.text
    results_page = client.get("/")
    assert 'class="collection-runner-row content-results-runner-row"' in results_page.text
    assert 'data-query-defaults-form' in results_page.text
    assert "old failure" not in page.text

    response = client.post(
        f"/papers/{paper_id}/analyze",
        data={"scope": "abstract", "model": "default"},
        headers={"X-Paperwatch-Async": "1"},
    )
    assert response.status_code == 200 and response.json()["batch_id"] == "analysis-web"
    response = client.post(
        f"/papers/{paper_id}/prompt",
        data={
            "query_id": query_id,
            "model": "missing",
            "source_scope": "abstract",
            "pdf_strategy": "auto",
            "context_window_tokens": 4096,
        },
        headers={"X-Paperwatch-Async": "1"},
    )
    assert response.status_code == 200 and response.json()["batch_id"] == "query-web"
    assert query.calls[-1][:3] == (query_id, ("default",), config.llm.context_window_tokens)
    assert query.calls[-1][3]["paper_ids"] == [paper_id]
    assert query.calls[-1][4:] == ("fulltext", "direct")

    response = client.post(
        f"/papers/{paper_id}/prompt",
        data={
            "query_id": query_id,
            "override_query_defaults": "1",
            "model": "default",
            "source_scope": "abstract",
            "pdf_strategy": "auto",
            "context_window_tokens": 4096,
        },
        headers={"X-Paperwatch-Async": "1"},
    )
    assert response.status_code == 200 and response.json()["batch_id"] == "query-web"
    assert query.calls[-1][:3] == (query_id, ("default",), 4096)
    assert query.calls[-1][4:] == ("abstract", "auto")
    assert client.post("/api/analysis-cancel").json()["ok"] is True
    assert client.post("/api/query-cancel").json()["ok"] is True

    script = client.get("/static/app.js").text
    assert 'event.preventDefault()' in script
    assert 'window.scrollTo(0, scrollY)' in script
    assert 'window.location.reload()' in script  # only the global fetch page, not paper jobs
    assert 'form[data-paper-job]' in script
    assert 'X-Paperwatch-Async' in script


def test_restart_marks_abandoned_attempts_and_hides_them(
    config_path, paper_record
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        db.create_analysis_tasks(
            batch_id="abandoned", paper_id=paper_id, scope="abstract",
            profiles=[("default", "ollama", "test-model")],
        )
    client = TestClient(
        create_app(
            config_path,
            run_controller=RunStub(),
            analysis_controller=AnalysisWebStub(),
            query_controller=QueryWebStub(),
            author_client_factory=lambda config: AuthorStub(),
        )
    )
    assert "abandoned" not in client.get(f"/papers/{paper_id}").text
    with Database(config.paths.database) as db:
        task = db.analysis_tasks_for_batch("abandoned")[0]
        assert task["status"] == "failed"
        assert "restarted" in task["error_text"]


def test_job_runtime_helpers():
    events = []
    publish(lambda kind, value: events.append((kind, value)), "message", "x")
    publish(None, "message", "ignored")
    assert events == [("message", "x")]
    event = threading.Event()
    check_cancelled(event)
    event.set()
    with pytest.raises(JobCancelled):
        check_cancelled(event)


def test_analysis_controller_validation_legacy_and_failure(config_path, paper_record):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)

    controller = AnalysisController(config_path, runner=lambda *args: {"completed": 0, "failed": 0})
    with pytest.raises(ValueError, match="scope"):
        controller.start(paper_id, "bad", ("default",))
    with pytest.raises(ValueError, match="strategy"):
        controller.start(paper_id, "abstract", ("default",), "bad")
    with pytest.raises(ValueError, match="only available"):
        controller.start(paper_id, "abstract", ("default",), "chunked")
    with pytest.raises(ValueError, match="Context window"):
        controller.start(paper_id, "abstract", ("default",), context_window_tokens=10)
    with pytest.raises(ValueError, match="Unknown"):
        controller.start(paper_id, "abstract", ("missing",))
    with pytest.raises(ValueError, match="at least one"):
        controller.start(paper_id, "abstract", ())
    with pytest.raises(ValueError, match="Paper not found"):
        controller.start(999999, "abstract", ("default",))

    called = []

    def legacy(path, batch, paper, scope, profiles):
        called.append((paper, scope, profiles))
        return {"completed": 1, "failed": 0}

    controller = AnalysisController(config_path, runner=legacy)
    assert controller.start(paper_id, "abstract", ("default",))
    assert controller.start(paper_id, "abstract", ("default",)) is None
    controller.wait()
    assert called == [(paper_id, "abstract", ("default",))]
    assert controller.snapshot().completed == 1

    def seven(path, batch, paper, scope, profiles, strategy, context):
        return {"completed": 0, "failed": 1}

    controller = AnalysisController(config_path, runner=seven)
    assert controller.start(paper_id, "abstract", ("default",), context_window_tokens=2048)
    controller.wait()
    assert controller.snapshot().failed == 1

    def broken(*args):
        raise RuntimeError("runner broke")

    controller = AnalysisController(config_path, runner=broken)
    assert controller.start(paper_id, "abstract", ("default",))
    controller.wait()
    assert controller.snapshot().error == "runner broke"
    with Database(config.paths.database) as db:
        task = db.analysis_tasks_for_batch(controller.snapshot().batch_id)[0]
        assert task["status"] == "failed"


def test_analysis_batch_missing_profile_and_midrun_cancel(
    config_path, paper_record, analysis_result, monkeypatch
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        db.create_analysis_tasks(
            batch_id="gone-profile", paper_id=paper_id, scope="abstract",
            profiles=[("gone", "ollama", "gone-model")],
        )
    result = run_analysis_batch(
        config_path, "gone-profile", paper_id, "abstract", ("gone",)
    )
    assert result == {"completed": 0, "failed": 1}

    with Database(config.paths.database) as db:
        db.create_analysis_tasks(
            batch_id="mid-cancel", paper_id=paper_id, scope="abstract",
            profiles=[("default", "ollama", "test-model")],
        )
    cancel = threading.Event()

    class CancellingLLM(RuntimeAnalysisLLM):
        def analyze(self, prompt):
            response = super().analyze(prompt)
            cancel.set()
            return response

    monkeypatch.setattr(
        "paperwatch.analysis_jobs.create_llm",
        lambda profile: CancellingLLM(analysis_result),
    )
    result = run_analysis_batch(
        config_path, "mid-cancel", paper_id, "abstract", ("default",),
        cancel_event=cancel,
    )
    assert result["cancelled"] is True
    with Database(config.paths.database) as db:
        assert db.analysis_tasks_for_batch("mid-cancel")[0]["status"] == "failed"


def test_controller_progress_caps_and_close_failures(config_path):
    analysis = AnalysisController(config_path)
    analysis._progress("stage", "s" * 300)
    analysis._progress("message", "m" * 1500)
    analysis._progress("output", "x" * 100_001)
    assert len(analysis.snapshot().stage) == 200
    assert len(analysis.snapshot().message) == 1000
    assert len(analysis.snapshot().live_output) == 100_000
    analysis._progress("reset_output", "")
    assert analysis.snapshot().live_output == ""

    class BrokenClose:
        def close(self):
            raise RuntimeError("close")

    analysis._snapshot.running = True
    analysis._active_llm = BrokenClose()
    assert analysis.cancel() is True

    query = QueryController(config_path)
    query._progress("stage", "s" * 300)
    query._progress("message", "m" * 1500)
    query._progress("output", "x" * 100_001)
    assert len(query.snapshot().stage) == 200
    assert len(query.snapshot().message) == 1000
    assert len(query.snapshot().live_output) == 100_000
    query._progress("reset_output", "")
    query._snapshot.running = True
    query._active_llm = BrokenClose()
    assert query.cancel() is True


def test_query_helpers_validation_compaction_and_merging(monkeypatch):
    from paperwatch.llm import StructuredJSONResponse
    from paperwatch.query_jobs import (
        _compact_document, _generate, _pack_documents, run_query_for_model,
    )

    class Invalid:
        def generate_json(self, *args, **kwargs):
            return StructuredJSONResponse(data={"bad": True}, usage={}, raw={})

    with pytest.raises(Exception, match="Invalid collection-query result"):
        _generate(Invalid(), "x")

    large = {"paper_id": 1, "title": "T" * 500, "abstract": "A" * 5000}
    assert _compact_document(large, 200) != large
    assert _compact_document(large, 2) in ({}, large)
    groups = _pack_documents([large, large], 250)
    assert sum(len(group) for group in groups) == 2

    class MergeLLM:
        provider = "ollama"
        model = "merge"

        def __init__(self):
            self.calls = 0

        def generate_json(self, prompt, *, schema, schema_name):
            self.calls += 1
            payload = {
                "title": f"Result {self.calls}",
                "summary": "Merged result",
                "items": [{"paper_id": self.calls, "title": "P", "output": "O"}],
            }
            return StructuredJSONResponse(
                data=payload,
                usage={"tokens": 2, "provider": "ollama"},
                raw=payload,
            )

        def close(self):
            pass

    empty, usage, chars = run_query_for_model(
        MergeLLM(), user_prompt="x", documents=[], context_window_tokens=2048,
        max_input_chars=2000,
    )
    assert empty.title == "No matching papers" and usage == {} and chars == 0

    docs = [
        {"paper_id": index, "title": f"Paper {index}", "abstract": "x" * 1200}
        for index in range(4)
    ]
    events = []
    result, usage, chars = run_query_for_model(
        MergeLLM(), user_prompt="compare", documents=docs,
        context_window_tokens=512, max_input_chars=2500,
        progress=lambda kind, value: events.append((kind, value)),
    )
    assert result.summary == "Merged result"
    assert usage["tokens"] >= 2 and usage["provider"] == "ollama"
    assert chars > 0 and any("Merging" in value for _, value in events)

    monkeypatch.setattr(
        "paperwatch.query_jobs._pack_documents",
        lambda documents, budget: [[item] for item in documents],
    )
    with pytest.raises(Exception, match="exceeds the selected context"):
        run_query_for_model(
            MergeLLM(), user_prompt="x", documents=docs,
            context_window_tokens=512, max_input_chars=2500,
        )


def test_query_batch_failure_cancel_and_controller_validation(
    config_path, paper_record, monkeypatch
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        query_id = db.create_saved_query(
            name="Q", source_type="papers", search_query="", prompt="P"
        )
        db.create_query_runs(
            batch_id="gone-query-profile", query_id=query_id,
            profiles=[("gone", "ollama", "gone")], context_window_tokens=2048,
            selection={"paper_ids": [paper_id]},
        )
    assert run_query_batch(
        config_path, "gone-query-profile", query_id, ("gone",), 2048
    ) == {"completed": 0, "failed": 1}

    with Database(config.paths.database) as db:
        db.create_query_runs(
            batch_id="cancel-query", query_id=query_id,
            profiles=[("default", "ollama", "test-model")], context_window_tokens=2048,
            selection={"paper_ids": [paper_id]},
        )
    event = threading.Event()
    event.set()
    assert run_query_batch(
        config_path, "cancel-query", query_id, ("default",), 2048,
        cancel_event=event,
    )["cancelled"] is True

    controller = QueryController(config_path, runner=lambda *args: {"completed": 0, "failed": 0})
    with pytest.raises(ValueError, match="Context window"):
        controller.start(query_id, ("default",), 10)
    with pytest.raises(ValueError, match="at least one"):
        controller.start(query_id, (), 2048)
    with pytest.raises(ValueError, match="Unknown"):
        controller.start(query_id, ("missing",), 2048)
    with pytest.raises(ValueError, match="not found"):
        controller.start(999999, ("default",), 2048)

    def legacy(path, batch, query, profiles, context):
        return {"completed": 1, "failed": 0}

    controller = QueryController(config_path, runner=legacy)
    assert controller.start(query_id, ("default",), 2048, selection={"paper_ids": ["bad"]})
    assert controller.start(query_id, ("default",), 2048) is None
    controller.wait()
    assert controller.snapshot().completed == 1 and controller.snapshot().paper_id is None

    def broken(*args):
        raise RuntimeError("query runner broke")

    controller = QueryController(config_path, runner=broken)
    assert controller.start(query_id, ("default",), 2048)
    controller.wait()
    assert controller.snapshot().error == "query runner broke"
    with Database(config.paths.database) as db:
        run = db.query_runs_for_batch(controller.snapshot().batch_id)[0]
        assert run["status"] == "failed"


def test_ollama_stream_error_fallback_and_openai_runtime(analysis_result):
    from paperwatch.llm import OpenAIClient

    calls = []

    def fallback_handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/api/chat":
            return httpx.Response(404, text="not found")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": analysis_result.model_dump_json()}}]},
        )

    ollama = OllamaClient(
        model="m", base_url="http://local", timeout_seconds=1,
        client=httpx.Client(transport=httpx.MockTransport(fallback_handler)),
    )
    output = []
    ollama.set_runtime(lambda kind, value: output.append((kind, value)), threading.Event())
    assert ollama.analyze("x").result.name == analysis_result.name
    assert calls == ["/api/chat", "/v1/chat/completions"]
    assert output

    for response, message in [
        (httpx.Response(500, text="broken"), "HTTP 500"),
        (httpx.Response(200, text="not-json\n"), "streaming response failed"),
        (httpx.Response(200, text="{}\n"), "no message content"),
    ]:
        client = httpx.Client(
            transport=httpx.MockTransport(lambda request, item=response: item)
        )
        llm = OllamaClient(model="m", base_url="http://local", timeout_seconds=1, client=client)
        llm.set_runtime(lambda kind, value: None, threading.Event())
        with pytest.raises(Exception, match=message):
            llm.analyze("x")

    events = []
    openai = OpenAIClient(
        model="gpt", base_url="https://api.test", api_key="key", timeout_seconds=1,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"output_text": analysis_result.model_dump_json()})
            )
        ),
    )
    openai.set_runtime(lambda kind, value: events.append((kind, value)), threading.Event())
    assert openai.analyze("x").result.name == analysis_result.name
    assert events and events[0][0] == "output"


def test_web_async_validation_and_busy_responses(config_path, paper_record):
    config = load_config(config_path)
    from dataclasses import replace
    no_abstract = replace(paper_record, abstract=None, pdf_url=None)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(no_abstract)
        query_id = db.create_saved_query(
            name="Prompt", source_type="papers", search_query="", prompt="P"
        )

    class BusyAnalysis(AnalysisWebStub):
        def start(self, *args, **kwargs):
            return None

        def cancel(self):
            return False

    class BusyQuery(QueryWebStub):
        def start(self, *args, **kwargs):
            return None

        def cancel(self):
            return False

    client = TestClient(create_app(
        config_path, run_controller=RunStub(), analysis_controller=BusyAnalysis(),
        query_controller=BusyQuery(), author_client_factory=lambda config: AuthorStub(),
    ))
    headers = {"X-Paperwatch-Async": "1"}
    assert client.post(
        f"/papers/{paper_id}/analyze", data={"scope": "abstract", "model": "default"},
        headers=headers,
    ).status_code == 400
    assert client.post(
        f"/papers/{paper_id}/analyze", data={"scope": "fulltext", "model": "default"},
        headers=headers,
    ).status_code == 400
    assert client.post(
        f"/papers/{paper_id}/prompt", data={"model": "default"}, headers=headers,
    ).status_code == 400
    assert client.post(
        f"/papers/{paper_id}/prompt",
        data={
            "query_id": query_id,
            "override_query_defaults": "1",
            "model": "missing",
        }, headers=headers,
    ).status_code == 400
    assert client.post("/api/analysis-cancel").status_code == 409
    assert client.post("/api/query-cancel").status_code == 409

    with Database(config.paths.database) as db:
        db._conn.execute("UPDATE papers SET abstract='A', pdf_url='https://x/p.pdf' WHERE id=?", (paper_id,))
        db._conn.commit()
    assert client.post(
        f"/papers/{paper_id}/analyze", data={"scope": "abstract", "model": "default"},
        headers=headers,
    ).status_code == 409
    assert client.post(
        f"/papers/{paper_id}/prompt",
        data={"query_id": query_id, "model": "default"}, headers=headers,
    ).status_code == 409
