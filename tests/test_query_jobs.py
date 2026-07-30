from __future__ import annotations

import threading
from dataclasses import replace

import pytest

from paperwatch.config import load_config
from paperwatch.db import Database
from paperwatch.llm import StructuredJSONResponse
from paperwatch.models import BatchQueryResult
from paperwatch.query_jobs import (
    QueryController,
    _pack_documents,
    collect_query_documents,
    run_query_batch,
    run_query_for_model,
)


class FakeQueryLLM:
    provider = "ollama"

    def __init__(self, model="query-model", *, fail=False):
        self.model = model
        self.fail = fail
        self.calls: list[str] = []
        self.closed = False

    def generate_json(self, prompt, *, schema, schema_name):
        self.calls.append(prompt)
        if self.fail:
            raise RuntimeError("query model failed")
        result = BatchQueryResult(
            title="Collection comparison",
            summary="The matching papers use related methods but differ in evidence and scope.",
            items=[
                {
                    "paper_id": 1,
                    "title": "Minimal Agents for Reliable Machine Learning",
                    "output": "The paper contributes a compact method with limited benchmark evidence.",
                }
            ],
        )
        return StructuredJSONResponse(data=result.model_dump(), usage={"eval_count": 4}, raw={})

    def close(self):
        self.closed = True


def _add_second_profile(config_path):
    with config_path.open("a", encoding="utf-8") as handle:
        handle.write(
            """

[[llm_profiles]]
name = "second"
provider = "ollama"
model = "second-model"
base_url = "http://second.test"
timeout_seconds = 1
max_input_chars = 5000
context_window_tokens = 2048
analyze_pdfs = true
prompt_version = "test-v1"
"""
        )


def test_collect_documents_for_papers_and_summaries(
    config_path, paper_record, analysis_result
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        db.save_analysis(
            paper_id=paper_id,
            scope="abstract",
            provider="ollama",
            model="m",
            prompt_version="v1",
            content_sha256="x",
            input_chars=10,
            result=analysis_result,
            usage={},
        )
        papers = collect_query_documents(db, "papers", "Minimal")
        assert papers[0]["paper_id"] == paper_id
        assert papers[0]["abstract"] == paper_record.abstract
        summaries = collect_query_documents(db, "summaries", "Minimal")
        assert summaries[0]["summary"]["paper_kind"] == "machine_learning"
        assert len(collect_query_documents(db, "summaries", "")) == 1
        assert collect_query_documents(db, "summaries", "limited benchmark set")
        assert collect_query_documents(db, "summaries", "missing") == []
        with pytest.raises(ValueError, match="papers or summaries"):
            collect_query_documents(db, "bad", "")


def test_collection_query_map_reduce_and_empty_result():
    llm = FakeQueryLLM()
    documents = [
        {"paper_id": index, "title": f"Paper {index}", "abstract": "detail " * 300}
        for index in range(1, 7)
    ]
    result, usage, input_chars = run_query_for_model(
        llm,
        user_prompt="Compare evidence",
        documents=documents,
        context_window_tokens=1024,
        max_input_chars=5000,
    )
    assert result.title == "Collection comparison"
    assert len(llm.calls) > 1
    assert usage["eval_count"] >= 4
    assert input_chars > 0

    empty, usage, chars = run_query_for_model(
        llm,
        user_prompt="x",
        documents=[],
        context_window_tokens=1024,
        max_input_chars=5000,
    )
    assert empty.items == []
    assert usage == {}
    assert chars == 0

    packed = _pack_documents(
        [{"title": "huge", "summary": "x" * 5000, "items": [{"title": "a", "output": "b"}]}],
        1000,
    )
    assert packed
    assert len(__import__("json").dumps(packed[0], ensure_ascii=False, separators=(",", ":"))) <= 1000
    assert packed[0][0]["title"] == "huge"

    nested = _pack_documents(
        [
            {
                "paper_id": 3,
                "title": "Nested output",
                "summary": {"items": [{"output": "\\theta " * 20_000}]},
            }
        ],
        800,
    )
    assert len(__import__("json").dumps(nested[0], ensure_ascii=False, separators=(",", ":"))) <= 800
    assert nested[0][0]["paper_id"] == 3


def test_query_batch_persists_success_and_failure(
    config_path, paper_record, monkeypatch
):
    _add_second_profile(config_path)
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.upsert_paper(paper_record)
        query_id = db.create_saved_query(
            name="Compare", source_type="papers", search_query="Minimal", prompt="Compare"
        )
        profiles = config.available_llms
        db.create_query_runs(
            batch_id="batch",
            query_id=query_id,
            profiles=[(p.profile, p.provider, p.model) for p in profiles],
            context_window_tokens=2048,
        )

    clients = []

    def factory(profile):
        client = FakeQueryLLM(profile.model, fail=profile.profile == "second")
        clients.append(client)
        return client

    monkeypatch.setattr("paperwatch.query_jobs.create_llm", factory)
    assert run_query_batch(
        config_path, "batch", query_id, ("default", "second"), 2048
    ) == {"completed": 1, "failed": 1}
    assert all(client.closed for client in clients)
    with Database(config.paths.database) as db:
        rows = db.query_runs_for_batch("batch")
        assert [row["status"] for row in rows] == ["success", "failed"]
        assert rows[0]["output_json"]
        assert "query model failed" in rows[1]["error_text"]
        assert db.get_saved_query(query_id)["last_run_at"]


def test_query_controller_validation_busy_and_failure(config_path, paper_record):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.upsert_paper(paper_record)
        query_id = db.create_saved_query(
            name="Q", source_type="papers", search_query="", prompt="Summarize"
        )
    gate = threading.Event()

    def runner(path, batch, query, profiles, context):
        gate.wait(1)
        return {"completed": len(profiles), "failed": 0}

    controller = QueryController(config_path, runner=runner)
    with pytest.raises(ValueError, match="512"):
        controller.start(query_id, ("default",), 100)
    with pytest.raises(ValueError, match="Select"):
        controller.start(query_id, (), 1024)
    with pytest.raises(ValueError, match="Unknown"):
        controller.start(query_id, ("missing",), 1024)
    batch = controller.start(query_id, ("default", "default"), 1024)
    assert batch
    assert controller.start(query_id, ("default",), 1024) is None
    gate.set()
    controller.wait()
    assert controller.snapshot().completed == 1

    broken = QueryController(
        config_path,
        runner=lambda *args: (_ for _ in ()).throw(RuntimeError("query boom")),
    )
    assert broken.start(query_id, ("default",), 1024)
    broken.wait()
    assert broken.snapshot().error == "query boom"
