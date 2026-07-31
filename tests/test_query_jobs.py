from __future__ import annotations

import threading
from dataclasses import replace

import pytest

from paperwatch.config import load_config
from paperwatch.db import Database
from paperwatch.llm import LLMError, StructuredJSONResponse
from paperwatch.models import BatchQueryResult
from paperwatch.query_jobs import (
    QueryController,
    _pack_documents,
    _selection_search_args,
    _split_oversized_document,
    collect_query_documents,
    run_query_batch,
    run_query_for_model,
    validate_query_sources,
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


def test_collect_documents_for_abstracts_and_selected_ids(
    config_path, paper_record, analysis_result
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        papers = collect_query_documents(db, "abstract", "Minimal")
        assert papers[0]["paper_id"] == paper_id
        assert papers[0]["abstract"] == paper_record.abstract
        assert papers[0]["source"] == "abstract"
        assert collect_query_documents(db, "papers", selection={"paper_ids": [paper_id]})
        assert collect_query_documents(db, "abstract", "missing") == []
        with pytest.raises(ValueError, match="abstract or fulltext"):
            collect_query_documents(db, "bad", "")


def test_collect_documents_uses_recent_page_selection(
    config_path, author_candidate, paper_record, analysis_result
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        paper_id, _ = db.upsert_paper(paper_record)
        selection = {
            "query": "Minimal",
            "author_id": "A123",
            "topic_id": "T100",
            "paper_kind": "",
            "scope": "",
            "has_pdf": True,
            "analyzed": None,
            "date_from": "2026-01-01",
            "date_to": "2026-12-31",
            "sort": "title",
        }
        papers = collect_query_documents(
            db, "abstract", "ignored legacy term", selection=selection
        )
        assert [item["paper_id"] for item in papers] == [paper_id]
        selected = collect_query_documents(
            db, "abstract", selection={"paper_ids": [paper_id, "bad", 999999]}
        )
        assert [item["paper_id"] for item in selected] == [paper_id]

    args = _selection_search_args({"has_pdf": "yes", "analyzed": 1})
    assert args["has_pdf"] is None
    assert args["analyzed"] is None
    assert _selection_search_args(None)["sort"] == "newest"


def test_saved_prompt_source_validation_checks_every_selected_paper(
    config_path, paper_record
):
    config = load_config(config_path)
    abstract_only = replace(
        paper_record,
        openalex_id="W-ABSTRACT-ONLY",
        doi="10.1000/abstract-only",
        arxiv_id=None,
        title="Abstract only paper",
        pdf_url=None,
    )
    pdf_only = replace(
        paper_record,
        openalex_id="W-PDF-ONLY",
        doi="10.1000/pdf-only",
        arxiv_id=None,
        title="PDF only paper",
        abstract=None,
        pdf_url="https://example.test/pdf-only.pdf",
    )
    local_only = replace(
        paper_record,
        openalex_id="W-LOCAL-ONLY",
        doi="10.1000/local-only",
        arxiv_id=None,
        title="Local PDF only paper",
        abstract=None,
        pdf_url=None,
    )
    stale_local = replace(
        paper_record,
        openalex_id="W-STALE-LOCAL",
        doi="10.1000/stale-local",
        arxiv_id=None,
        title="Stale local PDF paper",
        abstract=None,
        pdf_url=None,
    )
    with Database(config.paths.database) as db:
        abstract_id, _ = db.upsert_paper(abstract_only)
        pdf_id, _ = db.upsert_paper(pdf_only)
        local_id, _ = db.upsert_paper(local_only)
        stale_id, _ = db.upsert_paper(stale_local)
        local_path = config.paths.pdf_dir / "local-only.pdf"
        local_path.write_bytes(b"%PDF-1.4\n%%EOF")
        db.set_local_pdf(local_id, str(local_path))
        db.set_local_pdf(stale_id, str(config.paths.pdf_dir / "missing.pdf"))

        assert validate_query_sources(
            db,
            selection={"paper_ids": [abstract_id]},
            source_scope="abstract",
            pdf_strategy="auto",
        ) == 1
        assert validate_query_sources(
            db,
            selection={"paper_ids": [pdf_id]},
            source_scope="fulltext",
            pdf_strategy="direct",
        ) == 1
        assert validate_query_sources(
            db,
            selection={"paper_ids": [local_id]},
            source_scope="fulltext",
            pdf_strategy="chunked",
        ) == 1

        with pytest.raises(ValueError, match="PDF strategy only applies"):
            validate_query_sources(
                db,
                selection={"paper_ids": [abstract_id]},
                source_scope="abstract",
                pdf_strategy="chunked",
            )
        with pytest.raises(ValueError, match="1 selected paper.*abstract.*PDF only paper"):
            validate_query_sources(
                db,
                selection={"paper_ids": [abstract_id, pdf_id]},
                source_scope="abstract",
                pdf_strategy="auto",
            )
        with pytest.raises(ValueError, match="1 selected paper.*PDF.*Abstract only paper"):
            validate_query_sources(
                db,
                selection={"paper_ids": [abstract_id, pdf_id]},
                source_scope="fulltext",
                pdf_strategy="auto",
            )
        with pytest.raises(ValueError, match="Stale local PDF paper"):
            validate_query_sources(
                db,
                selection={"paper_ids": [stale_id]},
                source_scope="fulltext",
                pdf_strategy="auto",
            )
        with pytest.raises(ValueError, match="No papers match"):
            validate_query_sources(
                db,
                selection={"query": "definitely missing"},
                source_scope="abstract",
                pdf_strategy="auto",
            )


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


def test_query_batch_persisted_selection_and_edge_cases(
    config_path, paper_record, monkeypatch
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.upsert_paper(paper_record)
        query_id = db.create_saved_query(
            name="Filtered", source_type="papers", search_query="Minimal", prompt="Compare"
        )
        run_id = db.create_query_runs(
            batch_id="filtered",
            query_id=query_id,
            profiles=[("default", "ollama", "test-model")],
            context_window_tokens=2048,
            selection={"query": "does-not-match", "labels": ["No matches"]},
        )[0]

    client = FakeQueryLLM()
    monkeypatch.setattr("paperwatch.query_jobs.create_llm", lambda profile: client)
    assert run_query_batch(
        config_path, "filtered", query_id, ("default",), 2048
    ) == {"completed": 1, "failed": 0}
    assert client.calls == []
    with Database(config.paths.database) as db:
        row = db._conn.execute("SELECT * FROM query_runs WHERE id=?", (run_id,)).fetchone()
        assert row["matched_count"] == 0
        assert "No matching papers" in row["output_json"]

        malformed_id = db.create_query_runs(
            batch_id="malformed",
            query_id=query_id,
            profiles=[("default", "ollama", "test-model")],
            context_window_tokens=2048,
        )[0]
        db._conn.execute(
            "UPDATE query_runs SET selection_json='{bad' WHERE id=?", (malformed_id,)
        )
        db._conn.commit()
    client.calls.clear()
    assert run_query_batch(
        config_path, "malformed", query_id, ("default",), 2048
    )["completed"] == 1
    assert client.calls

    assert run_query_batch(
        config_path, "no-rows", query_id, ("default",), 2048
    ) == {"completed": 0, "failed": 0}
    with pytest.raises(ValueError, match="Saved query"):
        run_query_batch(config_path, "none", 999999, ("default",), 2048)

    with Database(config.paths.database) as db:
        removed_id = db.create_query_runs(
            batch_id="removed-profile",
            query_id=query_id,
            profiles=[("removed", "ollama", "gone")],
            context_window_tokens=2048,
        )[0]
    assert run_query_batch(
        config_path, "removed-profile", query_id, ("removed", "missing-row"), 2048
    ) == {"completed": 0, "failed": 1}
    with Database(config.paths.database) as db:
        assert "no longer available" in db._conn.execute(
            "SELECT error_text FROM query_runs WHERE id=?", (removed_id,)
        ).fetchone()[0]


def test_query_controller_validation_busy_and_failure(config_path, paper_record):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        abstract_only_id, _ = db.upsert_paper(
            replace(
                paper_record,
                openalex_id="W-NO-PDF",
                doi="10.1000/no-pdf",
                arxiv_id=None,
                title="Selected paper without a PDF",
                pdf_url=None,
            )
        )
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
    with pytest.raises(ValueError, match="abstract or fulltext"):
        controller.start(query_id, ("default",), 1024, source_scope="summary")
    with pytest.raises(ValueError, match="auto, direct, or chunked"):
        controller.start(query_id, ("default",), 1024, pdf_strategy="invalid")
    with pytest.raises(ValueError, match="Selected paper without a PDF"):
        controller.start(
            query_id,
            ("default",),
            1024,
            selection={"paper_ids": [paper_id, abstract_only_id]},
            source_scope="fulltext",
        )
    with Database(config.paths.database) as db:
        assert db.list_query_runs(query_id) == []
    selection = {"query": "Minimal", "has_pdf": True}
    batch = controller.start(
        query_id, ("default", "default"), 1024, selection=selection
    )
    assert batch
    with Database(config.paths.database) as db:
        stored = db.query_runs_for_batch(batch)[0]
        stored_selection = __import__("json").loads(stored["selection_json"])
        assert stored_selection == {
            **selection, "source_scope": "abstract", "pdf_strategy": "auto"
        }
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

    missing = QueryController(config_path, runner=runner)
    with pytest.raises(ValueError, match="Saved query"):
        missing.start(999999, ("default",), 1024)


def test_collect_fulltext_documents_closes_owned_client_and_reports_errors(
    config_path, paper_record, monkeypatch
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)

        class PDFClient:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        owned = PDFClient()
        monkeypatch.setattr("paperwatch.query_jobs.httpx.Client", lambda **kwargs: owned)
        seen = {}

        def paper_text(cfg, database, selected_id, scope, client):
            seen.update(
                cfg=cfg, database=database, selected_id=selected_id,
                scope=scope, client=client
            )
            return database.get_paper(selected_id), "Complete extracted PDF text"

        monkeypatch.setattr("paperwatch.query_jobs._paper_text", paper_text)
        documents = collect_query_documents(
            db, "fulltext", selection={"paper_ids": [paper_id]}, config=config
        )
        assert documents[0]["fulltext"] == "Complete extracted PDF text"
        assert documents[0]["source"] == "fulltext"
        assert seen["scope"] == "fulltext"
        assert seen["client"] is owned
        assert owned.closed

        with pytest.raises(ValueError, match="require application configuration"):
            collect_query_documents(
                db, "fulltext", selection={"paper_ids": [paper_id]}
            )

        def broken_text(*args):
            raise OSError("corrupt PDF")

        monkeypatch.setattr("paperwatch.query_jobs._paper_text", broken_text)
        with pytest.raises(ValueError, match="Could not load PDF.*corrupt PDF"):
            collect_query_documents(
                db, "fulltext", selection={"paper_ids": [paper_id]},
                config=config, pdf_client=PDFClient(),
            )


def test_pdf_query_strategies_and_oversized_document_splitting():
    llm = FakeQueryLLM()
    documents = [{"paper_id": 1, "title": "Small", "fulltext": "evidence"}]
    result, usage, chars = run_query_for_model(
        llm, user_prompt="Summarize", documents=documents,
        context_window_tokens=2048, max_input_chars=10000, pdf_strategy="direct",
    )
    assert result.title == "Collection comparison"
    assert usage["eval_count"] == 4
    assert chars > 0
    assert len(llm.calls) == 1

    with pytest.raises(ValueError, match="auto, direct, or chunked"):
        run_query_for_model(
            llm, user_prompt="x", documents=documents,
            context_window_tokens=2048, max_input_chars=10000, pdf_strategy="bad",
        )

    huge = [{"paper_id": 1, "title": "Huge", "fulltext": "x" * 20000}]
    with pytest.raises(LLMError, match="exceed the context window"):
        run_query_for_model(
            llm, user_prompt="x", documents=huge,
            context_window_tokens=512, max_input_chars=2000, pdf_strategy="direct",
        )

    chunks = _split_oversized_document(huge[0], 700)
    assert len(chunks) > 1
    assert {chunk.get("paper_id") for chunk in chunks} == {1}
    assert [chunk.get("chunk_index") for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert all(__import__("json").dumps([chunk], ensure_ascii=False).__len__() <= 730 for chunk in chunks)

    compact_only = _split_oversized_document(
        {"paper_id": 2, "title": "No text", "items": [{"output": "y" * 5000}]}, 500
    )
    assert len(compact_only) == 1


def test_query_batch_uses_runtime_source_and_pdf_strategy(
    config_path, paper_record, monkeypatch
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        query_id = db.create_saved_query(name="Runtime", prompt="Review")
        db.create_query_runs(
            batch_id="runtime", query_id=query_id,
            profiles=[("default", "ollama", "test-model")],
            context_window_tokens=4096,
            selection={
                "paper_ids": [paper_id], "source_scope": "fulltext",
                "pdf_strategy": "direct",
            },
        )

    captured = {}

    def collect(db, source_scope, **kwargs):
        captured["source_scope"] = source_scope
        captured["selection"] = kwargs["selection"]
        return [{"paper_id": paper_id, "title": "Runtime", "fulltext": "short"}]

    def run_model(llm, **kwargs):
        captured["pdf_strategy"] = kwargs["pdf_strategy"]
        return (
            BatchQueryResult(title="Done", summary="Done", items=[]),
            {},
            10,
        )

    client = FakeQueryLLM()
    monkeypatch.setattr("paperwatch.query_jobs.collect_query_documents", collect)
    monkeypatch.setattr("paperwatch.query_jobs.run_query_for_model", run_model)
    monkeypatch.setattr("paperwatch.query_jobs.create_llm", lambda profile: client)
    assert run_query_batch(
        config_path, "runtime", query_id, ("default",), 4096
    ) == {"completed": 1, "failed": 0}
    assert captured["source_scope"] == "fulltext"
    assert captured["pdf_strategy"] == "direct"
    assert captured["selection"]["paper_ids"] == [paper_id]
