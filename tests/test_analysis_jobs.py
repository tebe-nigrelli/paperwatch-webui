from __future__ import annotations

import threading
from pathlib import Path

import pytest

from paperwatch.analysis_jobs import (
    AnalysisController,
    _paper_text,
    run_analysis_batch,
)
from paperwatch.config import load_config
from paperwatch.db import Database
from paperwatch.models import LLMResponse


def add_second_profile(config_path: Path) -> None:
    with config_path.open("a", encoding="utf-8") as handle:
        handle.write(
            """

[[llm_profiles]]
name = "second-local"
provider = "ollama"
model = "test-model-2"
base_url = "http://ollama-two.test"
timeout_seconds = 1
max_input_chars = 5000
analyze_pdfs = true
prompt_version = "test-v1"
"""
        )


class FakeLLM:
    provider = "ollama"

    def __init__(self, model: str, result, calls: list[str], *, fail: bool = False):
        self.model = model
        self.result = result
        self.calls = calls
        self.fail = fail
        self.closed = False

    def analyze(self, prompt: str):
        self.calls.append(self.model)
        if self.fail:
            raise RuntimeError(f"{self.model} failed")
        return LLMResponse(result=self.result, usage={"model": self.model})

    def close(self):
        self.closed = True


def test_multi_model_abstract_batch_is_cached(
    config_path, paper_record, analysis_result, monkeypatch
):
    add_second_profile(config_path)
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        db.create_analysis_tasks(
            batch_id="first",
            paper_id=paper_id,
            scope="abstract",
            profiles=[
                (item.profile, item.provider, item.model) for item in config.available_llms
            ],
        )

    calls: list[str] = []
    clients: list[FakeLLM] = []

    def factory(profile):
        client = FakeLLM(profile.model, analysis_result, calls)
        clients.append(client)
        return client

    monkeypatch.setattr("paperwatch.analysis_jobs.create_llm", factory)
    assert run_analysis_batch(
        config_path, "first", paper_id, "abstract", ("default", "second-local")
    ) == {"completed": 2, "failed": 0}
    assert calls == ["test-model", "test-model-2"]
    assert all(client.closed for client in clients)

    with Database(config.paths.database) as db:
        assert len(db.paper_analyses(paper_id)) == 2
        assert {row["status"] for row in db.analysis_tasks_for_batch("first")} == {
            "success"
        }
        db.create_analysis_tasks(
            batch_id="second",
            paper_id=paper_id,
            scope="abstract",
            profiles=[
                (item.profile, item.provider, item.model) for item in config.available_llms
            ],
        )

    assert run_analysis_batch(
        config_path, "second", paper_id, "abstract", ("default", "second-local")
    ) == {"completed": 2, "failed": 0}
    assert calls == ["test-model", "test-model-2"]  # cache avoided model calls


def test_batch_keeps_success_when_one_model_fails(
    config_path, paper_record, analysis_result, monkeypatch
):
    add_second_profile(config_path)
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        db.create_analysis_tasks(
            batch_id="mixed",
            paper_id=paper_id,
            scope="abstract",
            profiles=[
                (item.profile, item.provider, item.model) for item in config.available_llms
            ],
        )

    monkeypatch.setattr(
        "paperwatch.analysis_jobs.create_llm",
        lambda profile: FakeLLM(
            profile.model,
            analysis_result,
            [],
            fail=profile.profile == "second-local",
        ),
    )
    assert run_analysis_batch(
        config_path, "mixed", paper_id, "abstract", ("default", "second-local")
    ) == {"completed": 1, "failed": 1}
    with Database(config.paths.database) as db:
        tasks = db.analysis_tasks_for_batch("mixed")
        assert [row["status"] for row in tasks] == ["success", "failed"]
        assert "test-model-2 failed" in tasks[1]["error_text"]


def test_paper_text_downloads_pdf_and_validates_sources(
    config_path, paper_record, monkeypatch, tmp_path
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)

        def fake_download(url, destination, **kwargs):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"%PDF-test")
            return True

        monkeypatch.setattr("paperwatch.analysis_jobs.download_pdf", fake_download)
        monkeypatch.setattr(
            "paperwatch.analysis_jobs.extract_pdf_text", lambda path, max_chars: "complete text"
        )
        paper, text = _paper_text(config, db, paper_id, "fulltext", object())
        assert paper.title == paper_record.title
        assert text == "complete text"
        assert Path(db.get_paper(paper_id)["local_pdf_path"]).is_file()
        assert _paper_text(config, db, paper_id, "abstract", object())[1] == paper_record.abstract
        with pytest.raises(ValueError, match="scope"):
            _paper_text(config, db, paper_id, "bad", object())
        with pytest.raises(ValueError, match="not found"):
            _paper_text(config, db, 99999, "abstract", object())


def test_batch_marks_all_failed_when_source_is_missing(
    config_path, paper_record, analysis_result
):
    from dataclasses import replace

    config = load_config(config_path)
    paper_record = replace(paper_record, abstract=None, pdf_url=None)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        db.create_analysis_tasks(
            batch_id="no-source",
            paper_id=paper_id,
            scope="fulltext",
            profiles=[("default", "ollama", "test-model")],
        )
    assert run_analysis_batch(
        config_path, "no-source", paper_id, "fulltext", ("default",)
    ) == {"completed": 0, "failed": 1}
    with Database(config.paths.database) as db:
        task = db.analysis_tasks_for_batch("no-source")[0]
        assert task["status"] == "failed"
        assert "no PDF" in task["error_text"]


def test_analysis_controller_busy_success_and_failure(
    config_path, paper_record
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)

    gate = threading.Event()

    def runner(path, batch, paper, scope, profiles):
        gate.wait(1)
        return {"completed": len(profiles), "failed": 0}

    controller = AnalysisController(config_path, runner=runner)
    with pytest.raises(ValueError, match="scope"):
        controller.start(paper_id, "bad", ("default",))
    with pytest.raises(ValueError, match="Unknown configured"):
        controller.start(paper_id, "abstract", ("missing",))
    batch = controller.start(paper_id, "abstract", ("default", "default"))
    assert batch
    assert controller.start(paper_id, "abstract", ("default",)) is None
    assert controller.snapshot().profiles == ("default",)
    gate.set()
    controller.wait()
    assert controller.snapshot().completed == 1
    assert not controller.snapshot().running

    broken = AnalysisController(
        config_path,
        runner=lambda *args: (_ for _ in ()).throw(RuntimeError("batch boom")),
    )
    assert broken.start(paper_id, "abstract", ("default",))
    broken.wait()
    assert broken.snapshot().error == "batch boom"
    with Database(config.paths.database) as db:
        assert db.analysis_tasks_for_paper(paper_id)[0]["status"] == "failed"


def test_client_construction_failure_is_isolated_per_model(
    config_path, paper_record, analysis_result, monkeypatch
):
    add_second_profile(config_path)
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        db.create_analysis_tasks(
            batch_id="client-failure",
            paper_id=paper_id,
            scope="abstract",
            profiles=[
                (item.profile, item.provider, item.model) for item in config.available_llms
            ],
        )

    def factory(profile):
        if profile.profile == "default":
            raise RuntimeError("cannot create client")
        return FakeLLM(profile.model, analysis_result, [])

    monkeypatch.setattr("paperwatch.analysis_jobs.create_llm", factory)
    assert run_analysis_batch(
        config_path,
        "client-failure",
        paper_id,
        "abstract",
        ("default", "second-local"),
    ) == {"completed": 1, "failed": 1}
    with Database(config.paths.database) as db:
        tasks = db.analysis_tasks_for_batch("client-failure")
        assert tasks[0]["status"] == "failed"
        assert "cannot create client" in tasks[0]["error_text"]
        assert tasks[1]["status"] == "success"


def test_analysis_controller_chunk_options_and_task_metadata(config_path, paper_record):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
    controller = AnalysisController(config_path, runner=lambda *args: {"completed": 1, "failed": 0})
    with pytest.raises(ValueError, match="strategy"):
        controller.start(paper_id, "fulltext", ("default",), "bad")
    with pytest.raises(ValueError, match="only available"):
        controller.start(paper_id, "abstract", ("default",), "chunked")
    with pytest.raises(ValueError, match="512"):
        controller.start(paper_id, "fulltext", ("default",), "chunked", 100)
    batch = controller.start(paper_id, "fulltext", ("default",), "chunked", 2048)
    controller.wait()
    assert batch
    with Database(config.paths.database) as db:
        task = db.analysis_tasks_for_batch(batch)[0]
        assert task["analysis_strategy"] == "chunked"
        assert task["context_window_tokens"] == 2048


def test_run_analysis_batch_chunked_and_auto_paths(
    config_path, paper_record, analysis_result, monkeypatch
):
    from paperwatch.analysis import AnalysisOutcome
    from paperwatch.chunking import ChunkedAnalysisOutcome

    config = load_config(config_path)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        db.create_analysis_tasks(
            batch_id="chunked",
            paper_id=paper_id,
            scope="fulltext",
            profiles=[("default", "ollama", "test-model")],
            analysis_strategy="chunked",
            context_window_tokens=2048,
        )
    monkeypatch.setattr(
        "paperwatch.analysis_jobs._paper_text",
        lambda *args: (paper_record, "complete pdf text"),
    )
    fake = FakeLLM("test-model", analysis_result, [])
    monkeypatch.setattr("paperwatch.analysis_jobs.create_llm", lambda profile: fake)

    def chunked(**kwargs):
        analysis_id = kwargs["db"].save_analysis(
            paper_id=paper_id,
            scope="fulltext",
            provider="ollama",
            model="test-model",
            prompt_version="test-v1",
            content_sha256="chunked-hash",
            input_chars=20,
            result=analysis_result,
            usage={},
            analysis_strategy="chunked",
            context_window_tokens=2048,
            chunk_count=3,
        )
        return ChunkedAnalysisOutcome(
            AnalysisOutcome(analysis_result, analysis_id, True), 3
        )

    monkeypatch.setattr("paperwatch.analysis_jobs.analyze_chunked_text", chunked)
    assert run_analysis_batch(
        config_path, "chunked", paper_id, "fulltext", ("default",), "chunked", 2048
    ) == {"completed": 1, "failed": 0}
    with Database(config.paths.database) as db:
        assert db.analysis_tasks_for_batch("chunked")[0]["chunk_count"] == 3

    with Database(config.paths.database) as db:
        db.create_analysis_tasks(
            batch_id="auto",
            paper_id=paper_id,
            scope="fulltext",
            profiles=[("default", "ollama", "test-model")],
            analysis_strategy="auto",
        )
    assert run_analysis_batch(
        config_path, "auto", paper_id, "fulltext", ("default",), "auto", None
    )["completed"] == 1


def test_direct_analysis_rejects_context_that_cannot_hold_prompt(
    config_path, paper_record, analysis_result, monkeypatch
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        db.create_analysis_tasks(
            batch_id="tiny-context",
            paper_id=paper_id,
            scope="fulltext",
            profiles=[("default", "ollama", "test-model")],
            analysis_strategy="direct",
            context_window_tokens=512,
        )
    monkeypatch.setattr(
        "paperwatch.analysis_jobs._paper_text",
        lambda *args: (paper_record, "x" * 2000),
    )
    monkeypatch.setattr(
        "paperwatch.analysis_jobs.create_llm",
        lambda profile: FakeLLM(profile.model, analysis_result, []),
    )
    assert run_analysis_batch(
        config_path,
        "tiny-context",
        paper_id,
        "fulltext",
        ("default",),
        "direct",
        512,
    ) == {"completed": 0, "failed": 1}
    with Database(config.paths.database) as db:
        error = db.analysis_tasks_for_batch("tiny-context")[0]["error_text"]
        assert "use chunked analysis" in error


def test_codex_fulltext_batch_uploads_original_pdf_without_extracting_text(
    config_path, paper_record, analysis_result, monkeypatch
):
    with config_path.open("a", encoding="utf-8") as handle:
        handle.write(
            """

[[llm_profiles]]
name = "codex-review"
provider = "codex"
model = "gpt-5.3-codex"
api_key = "secret"
reasoning_effort = "high"
timeout_seconds = 1
max_input_chars = 5000
context_window_tokens = 400000
analyze_pdfs = true
prompt_version = "test-v1"
"""
        )
    config = load_config(config_path)
    pdf = config.paths.pdf_dir / "complete.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-1.7\nTHE ORIGINAL COMPLETE PAPER\n%%EOF")
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        db.set_local_pdf(paper_id, str(pdf))
        db.create_analysis_tasks(
            batch_id="codex-pdf",
            paper_id=paper_id,
            scope="fulltext",
            profiles=[("codex-review", "codex", "gpt-5.3-codex")],
            analysis_strategy="auto",
            context_window_tokens=400000,
        )

    uploaded = []

    class FakeCodex:
        provider = "codex"
        model = "gpt-5.3-codex"

        def analyze_pdf(self, prompt, pdf_path):
            uploaded.append((prompt, Path(pdf_path).read_bytes()))
            return LLMResponse(result=analysis_result, usage={"input_tokens": 10})

        def close(self):
            pass

    monkeypatch.setattr("paperwatch.analysis_jobs.create_llm", lambda profile: FakeCodex())
    monkeypatch.setattr(
        "paperwatch.analysis_jobs.extract_pdf_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not extract")),
    )
    assert run_analysis_batch(
        config_path,
        "codex-pdf",
        paper_id,
        "fulltext",
        ("codex-review",),
        "auto",
        400000,
    ) == {"completed": 1, "failed": 0}
    assert uploaded[0][1] == pdf.read_bytes()
    assert "attached as a PDF" in uploaded[0][0]
    with Database(config.paths.database) as db:
        analysis = db.paper_analyses(paper_id)[0]
        assert analysis["provider"] == "codex"
        assert analysis["analysis_strategy"] == "pdf_upload"
        assert analysis["input_chars"] > 0
