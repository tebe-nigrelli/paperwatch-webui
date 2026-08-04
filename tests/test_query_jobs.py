from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

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
    run_query_for_native_pdfs,
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


class FakeNativePDFLLM(FakeQueryLLM):
    provider = "codex"

    def __init__(self):
        super().__init__(model="native-query-model")
        self.pdf_calls: list[tuple[str, Path]] = []

    def generate_json_with_pdf(self, prompt, path, *, schema, schema_name):
        self.pdf_calls.append((prompt, path))
        index = len(self.pdf_calls)
        result = BatchQueryResult(
            title=f"Native result {index}",
            summary=f"Summary {index}",
            items=[
                {
                    "paper_id": index,
                    "title": f"Paper {index}",
                    "output": f"Output {index}",
                }
            ],
        )
        usage = (
            {"input_tokens": 2, "provider": "codex"}
            if index == 1
            else {"input_tokens": 3, "provider": "ignored", "cached": True}
        )
        return SimpleNamespace(data=result.model_dump(), usage=usage)


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


def _add_codex_profile(config_path):
    with config_path.open("a", encoding="utf-8") as handle:
        handle.write(
            """

[[llm_profiles]]
name = "codex-review"
provider = "codex"
model = "gpt-5.3-codex"
api_key = "secret"
timeout_seconds = 1
max_input_chars = 5000
context_window_tokens = 400000
analyze_pdfs = true
pdf_analysis_strategy = "direct"
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
        db.add_author_tag("A123", "theory")
        paper_id, _ = db.upsert_paper(paper_record)
        selection = {
            "query": "Minimal",
            "author_id": "A123",
            "author_tags": ["theory", ""],
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
    assert args["author_tags"] == ()
    assert _selection_search_args({"author_tags": "theory"})["author_tags"] == ()
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


def test_native_pdf_query_single_multi_and_provider_validation(monkeypatch):
    import paperwatch.query_jobs as query_jobs

    rows = [
        {"id": 1, "title": "Paper 1"},
        {"id": 2, "title": "Paper 2"},
    ]
    monkeypatch.setattr(
        query_jobs,
        "_paper_pdf_path",
        lambda config, db, paper_id, client: (None, Path(f"/tmp/{paper_id}.pdf")),
    )
    monkeypatch.setattr(
        query_jobs,
        "_paper_document",
        lambda db, row: {
            "paper_id": row["id"],
            "title": row["title"],
            "fulltext": "large text that must not be repeated in metadata",
        },
    )

    llm = FakeNativePDFLLM()
    single, usage, chars = run_query_for_native_pdfs(
        llm,
        user_prompt="Critique",
        papers=rows[:1],
        db=None,
        config=None,
        context_window_tokens=4096,
        max_input_chars=10_000,
    )
    assert single.title == "Native result 1"
    assert usage == {"input_tokens": 2, "provider": "codex"}
    assert chars > 0
    assert "large text" not in llm.pdf_calls[0][0]
    assert llm.pdf_calls[0][1] == Path("/tmp/1.pdf")

    merged = BatchQueryResult(title="Merged", summary="Merged summary", items=[])
    monkeypatch.setattr(
        query_jobs,
        "run_query_for_model",
        lambda *args, **kwargs: (
            merged,
            {"input_tokens": 7, "merge_provider": "ollama"},
            11,
        ),
    )
    llm = FakeNativePDFLLM()
    result, usage, chars = run_query_for_native_pdfs(
        llm,
        user_prompt="Compare",
        papers=rows,
        db=None,
        config=None,
        context_window_tokens=4096,
        max_input_chars=10_000,
    )
    assert result == merged
    assert usage == {
        "input_tokens": 12,
        "provider": "codex",
        "cached": True,
        "merge_provider": "ollama",
    }
    assert chars > 11

    with pytest.raises(LLMError, match="cannot read PDFs directly"):
        run_query_for_native_pdfs(
            FakeQueryLLM(),
            user_prompt="Compare",
            papers=rows,
            db=None,
            config=None,
            context_window_tokens=4096,
            max_input_chars=10_000,
        )


def test_native_pdf_batch_and_controller_force_fulltext(
    config_path, paper_record, monkeypatch
):
    import paperwatch.query_jobs as query_jobs

    _add_codex_profile(config_path)
    config = load_config(config_path)
    pdf = config.paths.pdf_dir / "native.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-1.7\ncomplete paper\n%%EOF")
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        db.set_local_pdf(paper_id, str(pdf))
        query_id = db.create_saved_query(
            name="Native critique", source_type="papers", search_query="", prompt="Critique"
        )
        db.create_query_runs(
            batch_id="native-batch",
            query_id=query_id,
            profiles=[("codex-review", "codex", "gpt-5.3-codex")],
            context_window_tokens=400000,
            selection={
                "paper_ids": [paper_id],
                "source_scope": "fulltext",
                "pdf_strategy": "direct",
            },
        )

    class NativeClient:
        provider = "codex"
        model = "gpt-5.3-codex"

        def close(self):
            self.closed = True

    client = NativeClient()
    client.closed = False
    active: list[object | None] = []
    seen: dict[str, object] = {}
    monkeypatch.setattr(query_jobs, "create_llm", lambda profile: client)

    def fake_native(llm, **kwargs):
        seen.update(kwargs)
        return (
            BatchQueryResult(
                title="Native batch result",
                summary="Native summary",
                items=[
                    {
                        "paper_id": paper_id,
                        "title": paper_record.title,
                        "output": "Native output",
                    }
                ],
            ),
            {"input_tokens": 4},
            321,
        )

    monkeypatch.setattr(query_jobs, "run_query_for_native_pdfs", fake_native)
    outcome = run_query_batch(
        config_path,
        "native-batch",
        query_id,
        ("codex-review",),
        2048,
        active_client=active.append,
    )
    assert outcome == {"completed": 1, "failed": 0}
    assert [row["id"] for row in seen["papers"]] == [paper_id]
    assert seen["context_window_tokens"] == 400000
    assert active == [client, None]
    assert client.closed is True
    with Database(config.paths.database) as db:
        row = db.query_runs_for_batch("native-batch")[0]
        assert row["status"] == "success" and row["input_chars"] == 321

    runner_calls = []

    def runner(path, batch, query, profiles, context):
        runner_calls.append((batch, query, profiles, context))
        return {"completed": 1, "failed": 0}

    controller = QueryController(config_path, runner=runner)
    batch_id = controller.start(
        query_id,
        ("codex-review",),
        2048,
        selection={"paper_ids": [paper_id]},
        source_scope="abstract",
        pdf_strategy="auto",
    )
    assert batch_id
    controller.wait()
    assert runner_calls[0][3] == 400000
    with Database(config.paths.database) as db:
        row = db.query_runs_for_batch(batch_id)[0]
        selection = __import__("json").loads(row["selection_json"])
        assert selection["source_scope"] == "fulltext"
        assert selection["pdf_strategy"] == "direct"


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


def test_blog_query_documents_validation_batch_and_controller(
    config_path, author_candidate, monkeypatch
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        blog_id = db.add_author_blog(
            author_candidate.openalex_id,
            "https://example.test/blog",
            label="Ada blog",
        )
        baseline_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Baseline",
            content_sha256="baseline",
            normalized_text="old text",
            diff_text="",
            change_kind="baseline",
            previous_entry_id=None,
        )
        entry_id = db.save_blog_entry(
            blog_id=blog_id,
            title="New systems note",
            content_sha256="changed",
            normalized_text="A detailed note about efficient inference.",
            diff_text="-old\n+new",
            change_kind="changed",
            previous_entry_id=baseline_id,
            entry_kind="rss",
            external_id="post-1",
            source_url="https://example.test/blog/post-1",
            published_at="2026-07-30T10:00:00+00:00",
        )
        documents = collect_query_documents(
            db,
            "blog",
            selection={
                "content_type": "blogs",
                "blog_entry_ids": [entry_id, "bad", -1, entry_id],
            },
        )
        assert len(documents) == 1
        assert documents[0]["blog_entry_id"] == entry_id
        assert documents[0]["content_type"] == "blog"
        assert documents[0]["authors"] == [author_candidate.name]
        assert "efficient inference" in documents[0]["content"]
        assert documents[0]["diff"] == "-old\n+new"
        assert db.blog_entries_by_ids(["bad", entry_id, entry_id, -1])[0]["id"] == entry_id

        selection = {"content_type": "blogs", "blog_entry_ids": [entry_id]}
        assert validate_query_sources(
            db, selection=selection, source_scope="blog", pdf_strategy="auto"
        ) == 1
        with pytest.raises(ValueError, match="stored blog entry text"):
            validate_query_sources(
                db, selection=selection, source_scope="abstract", pdf_strategy="auto"
            )
        with pytest.raises(ValueError, match="does not apply"):
            validate_query_sources(
                db, selection=selection, source_scope="blog", pdf_strategy="direct"
            )
        with pytest.raises(ValueError, match="No blog entries"):
            validate_query_sources(
                db,
                selection={"content_type": "blogs", "blog_entry_ids": [999999]},
                source_scope="blog",
                pdf_strategy="auto",
            )
        with pytest.raises(ValueError, match="papers or blogs"):
            collect_query_documents(db, selection={"content_type": "videos"})
        with pytest.raises(ValueError, match="Blog query source"):
            collect_query_documents(db, "abstract", selection=selection)

        query_id = db.create_saved_query(name="Blog summary", prompt="Summarize the post")
        db.create_query_runs(
            batch_id="blog-batch",
            query_id=query_id,
            profiles=[("default", "ollama", "test-model")],
            context_window_tokens=2048,
            selection={
                **selection,
                "source_scope": "blog",
                "pdf_strategy": "auto",
            },
        )

    client = FakeQueryLLM()
    monkeypatch.setattr("paperwatch.query_jobs.create_llm", lambda profile: client)
    result = run_query_batch(
        config_path, "blog-batch", query_id, ("default",), 2048
    )
    assert result == {"completed": 1, "failed": 0}
    assert "efficient inference" in client.calls[0]
    with Database(config.paths.database) as db:
        assert db.query_runs_for_blog_entry(entry_id)[0]["status"] == "success"

    gate = threading.Event()

    def runner(path, batch, query, profiles, context):
        gate.wait(1)
        return {"completed": 1, "failed": 0}

    controller = QueryController(config_path, runner=runner)
    batch_id = controller.start(
        query_id,
        ("default", "default"),
        2048,
        selection={"content_type": "blogs", "blog_entry_ids": [entry_id]},
        source_scope="abstract",
        pdf_strategy="chunked",
        content_type="blogs",
    )
    assert batch_id
    snapshot = controller.snapshot()
    assert snapshot.blog_entry_id == entry_id
    assert snapshot.content_type == "blogs"
    with Database(config.paths.database) as db:
        stored = db.query_runs_for_batch(batch_id)[0]
        selection_json = __import__("json").loads(stored["selection_json"])
        assert selection_json["source_scope"] == "blog"
        assert selection_json["pdf_strategy"] == "auto"
    gate.set()
    controller.wait()
    with pytest.raises(ValueError, match="papers or blogs"):
        controller.start(
            query_id,
            ("default",),
            2048,
            selection={"blog_entry_ids": [entry_id]},
            content_type="videos",
        )

def test_run_query_for_model_rejects_unknown_content_type():
    with pytest.raises(ValueError, match="papers or blogs"):
        run_query_for_model(
            FakeQueryLLM(),
            user_prompt="x",
            documents=[],
            context_window_tokens=2048,
            max_input_chars=5000,
            content_type="videos",
        )

def test_blog_query_records_without_diffs_and_run_lookup(config_path, author_candidate):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        blog_id = db.add_author_blog(author_candidate.openalex_id, "https://example.test/feed")
        entry_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Feed item",
            content_sha256="feed-item",
            normalized_text="Complete feed item body",
            diff_text="",
            change_kind="changed",
            previous_entry_id=None,
            entry_kind="rss",
            external_id="item-1",
        )
        documents = collect_query_documents(db, "blog", selection={"content_type": "blogs"})
        assert documents == [
            {
                "paper_id": None,
                "blog_entry_id": entry_id,
                "content_type": "blog",
                "title": "Feed item",
                "publication_date": documents[0]["publication_date"],
                "authors": [author_candidate.name],
                "blog": "Feed item",
                "entry_kind": "rss",
                "change_kind": "changed",
                "source_url": "https://example.test/feed",
                "content": "Complete feed item body",
                "diff": None,
            }
        ]
        assert [row["id"] for row in db.blog_entries_by_ids([None, "bad", -1, entry_id, entry_id])] == [entry_id]
        with pytest.raises(ValueError, match="Content type"):
            validate_query_sources(
                db,
                selection={"content_type": "videos"},
                source_scope="abstract",
                pdf_strategy="auto",
            )

        query_id = db.create_saved_query(name="Lookup", prompt="Review")
        run_id = db.create_query_runs(
            batch_id="blog-success",
            query_id=query_id,
            profiles=[("default", "ollama", "test-model")],
            context_window_tokens=2048,
            selection={"content_type": "blogs", "blog_entry_ids": [entry_id]},
        )[0]
        db.start_query_run(run_id)
        db.finish_query_run(
            run_id,
            matched_count=1,
            input_chars=10,
            output={"title": "Done", "summary": "Done", "items": []},
        )
        failed_id = db.create_query_runs(
            batch_id="blog-failed",
            query_id=query_id,
            profiles=[("default", "ollama", "test-model")],
            context_window_tokens=2048,
            selection={"content_type": "blogs", "blog_entry_ids": [str(entry_id)]},
        )[0]
        db.finish_query_run(failed_id, matched_count=0, input_chars=0, error="failed")
        malformed_id = db.create_query_runs(
            batch_id="blog-malformed",
            query_id=query_id,
            profiles=[("default", "ollama", "test-model")],
            context_window_tokens=2048,
        )[0]
        db._conn.execute(
            "UPDATE query_runs SET selection_json='{bad' WHERE id=?", (malformed_id,)
        )
        db._conn.commit()
        assert [row["id"] for row in db.query_runs_for_blog_entry(entry_id)] == [run_id]
        assert {row["id"] for row in db.query_runs_for_blog_entry(entry_id, successful_only=False)} == {
            run_id,
            failed_id,
        }

    llm = FakeQueryLLM()
    result, _usage, chars = run_query_for_model(
        llm,
        user_prompt="Summarize",
        documents=[
            {
                "blog_entry_id": entry_id,
                "content_type": "blog",
                "title": "Large blog",
                "content": "applied ML evidence " * 1500,
            }
        ],
        context_window_tokens=1024,
        max_input_chars=4000,
        content_type="blogs",
    )
    assert result.title == "Collection comparison"
    assert chars > 0
    assert len(llm.calls) > 1

def test_query_source_edge_cases_cover_blog_fallback_and_auto_missing(
    config_path, author_candidate, paper_record
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        blog_id = db.add_author_blog(author_candidate.openalex_id, "https://example.test/edge")
        baseline = db.save_blog_entry(
            blog_id=blog_id,
            title="Baseline",
            content_sha256="edge-base",
            normalized_text="base",
            diff_text="",
            change_kind="baseline",
            previous_entry_id=None,
        )
        entry_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Edge entry",
            content_sha256="edge-new",
            normalized_text="edge content",
            diff_text="+edge content",
            change_kind="changed",
            previous_entry_id=baseline,
        )
        all_blog_docs = collect_query_documents(
            db,
            "blog",
            selection={"content_type": "blogs", "blog_entry_ids": ["invalid"]},
        )
        assert [item["blog_entry_id"] for item in all_blog_docs] == [entry_id]

        paper_without_source = replace(
            paper_record,
            openalex_id="W-NO-SOURCE",
            doi="10.1000/no-source",
            arxiv_id=None,
            title="No source paper",
            abstract=None,
            pdf_url=None,
        )
        missing_id, _ = db.upsert_paper(paper_without_source)
        with pytest.raises(ValueError, match="abstract or PDF"):
            validate_query_sources(
                db,
                selection={"paper_ids": [missing_id]},
                source_scope="auto",
                pdf_strategy="auto",
            )
        with pytest.raises(ValueError, match="papers or blogs"):
            validate_query_sources(
                db,
                selection={"content_type": "videos"},
                source_scope="abstract",
                pdf_strategy="auto",
            )


def test_query_controller_tracks_numeric_progress_and_ignores_invalid(config_path):
    controller = QueryController(config_path, runner=lambda *args: {"completed": 0, "failed": 0})
    controller._progress("stage", "Running prompt")
    controller._progress("message", "Model one")
    controller._progress("reset_output", "")
    controller._progress("output", "partial")
    controller._progress("progress", "3/8")
    controller._progress("progress", "bad/value")
    snapshot = controller.snapshot()
    assert snapshot.stage == "Running prompt"
    assert snapshot.message == "Model one"
    assert snapshot.live_output == "partial"
    assert (snapshot.progress_current, snapshot.progress_total) == (3, 8)


def test_collection_query_accepts_outputs_longer_than_paper_summary_limit():
    from paperwatch.query_jobs import _generate

    long_output = "word " * 350

    class LongCollectionOutput:
        def generate_json(self, *args, **kwargs):
            payload = {
                "title": "Detailed collection result",
                "summary": long_output,
                "items": [
                    {
                        "paper_id": 1,
                        "blog_entry_id": None,
                        "content_type": "paper",
                        "title": "Detailed paper",
                        "output": long_output,
                    }
                ],
            }
            return StructuredJSONResponse(data=payload, usage={}, raw=payload)

    result, usage = _generate(LongCollectionOutput(), "prompt")
    assert len(result.summary.split()) == 350
    assert len(result.items[0].output.split()) == 350
    assert usage == {}


def _seed_multi_blog_queries(config_path, author_candidate):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        blog_id = db.add_author_blog(
            author_candidate.openalex_id,
            "https://example.test/multi-query-blog",
            label="Multi query blog",
        )
        entry_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Stored batch entry",
            content_sha256="multi-query-entry",
            normalized_text="Stored batch entry discusses efficient inference.",
            diff_text="+Stored batch entry discusses efficient inference.",
            change_kind="changed",
            previous_entry_id=None,
            source_url="https://example.test/multi-query-blog/post",
        )
        normal = db.create_saved_query(
            name="Normal blog query",
            prompt="Summarize this entry.",
            content_scope="blogs",
        )
        local = db.create_saved_query(
            name="Local regexp query",
            prompt="unused",
            content_scope="blogs",
            regexp_pattern=r"efficient\s+inference",
            regexp_mode="print",
        )
        papers = db.create_saved_query(
            name="Paper-only query",
            prompt="paper",
            content_scope="papers",
        )
    return entry_id, normal, local, papers


def test_query_controller_runs_multiple_blog_queries_sequentially(
    config_path, author_candidate
):
    entry_id, normal, local, _papers = _seed_multi_blog_queries(
        config_path, author_candidate
    )
    calls = []

    def runner(path, batch, query, profiles, context, progress, cancel, set_llm):
        calls.append((batch, query, profiles, context))
        progress("stage", f"Running {query}")
        progress("progress", "1/1")
        return {
            "completed": 1 if query == normal else 0,
            "failed": 0 if query == normal else 1,
        }

    controller = QueryController(config_path, runner=runner)
    batch_ids = controller.start_many(
        (normal, local, normal),
        "default",
        4096,
        selection={"query": "Stored", "blog_entry_ids": [entry_id]},
    )
    assert batch_ids is not None and len(batch_ids) == 2
    controller.wait()
    snapshot = controller.snapshot()
    assert snapshot.running is False
    assert snapshot.stage == "Complete"
    assert snapshot.completed == 1
    assert snapshot.failed == 1
    assert snapshot.progress_current == 2
    assert [item[1] for item in calls] == [normal, local]
    assert calls[0][2] == ("default",)
    assert calls[1][2] == ("regexp",)
    with Database(load_config(config_path).paths.database) as db:
        rows = [db.query_runs_for_batch(batch_id)[0] for batch_id in batch_ids]
        assert [row["profile"] for row in rows] == ["default", "regexp"]
        selection = __import__("json").loads(rows[0]["selection_json"])
        assert selection["content_type"] == "blogs"
        assert selection["source_scope"] == "blog"
        assert selection["pdf_strategy"] == "auto"


def test_query_controller_multi_query_validation_and_busy(
    config_path, author_candidate
):
    _entry_id, normal, _local, papers = _seed_multi_blog_queries(
        config_path, author_candidate
    )
    gate = threading.Event()

    def runner(path, batch, query, profiles, context):
        gate.wait(1)
        return {"completed": 1, "failed": 0}

    controller = QueryController(config_path, runner=runner)
    with pytest.raises(ValueError, match="between 512"):
        controller.start_many((normal,), "default", 100)
    with pytest.raises(ValueError, match="at least one"):
        controller.start_many((), "default", 2048)
    with pytest.raises(ValueError, match="blog results"):
        controller.start_many((normal,), "default", 2048, content_type="papers")
    with pytest.raises(ValueError, match="was not found"):
        controller.start_many((999999,), "default", 2048)
    with pytest.raises(ValueError, match="papers only"):
        controller.start_many((papers,), "default", 2048)
    with pytest.raises(ValueError, match="Unknown configured model"):
        controller.start_many((normal,), "missing", 2048)

    assert controller.start_many((normal,), "default", 2048)
    assert controller.start_many((normal,), "default", 2048) is None
    gate.set()
    controller.wait()
    assert controller.snapshot().stage == "Complete"


def test_query_controller_multi_query_cancel_and_failure(
    config_path, author_candidate
):
    _entry_id, normal, local, _papers = _seed_multi_blog_queries(
        config_path, author_candidate
    )
    cancelling = None
    call_count = 0

    def cancel_runner(path, batch, query, profiles, context):
        nonlocal call_count
        call_count += 1
        assert cancelling is not None
        cancelling._cancel_event.set()
        return {"completed": 1, "failed": 0}

    cancelling = QueryController(config_path, runner=cancel_runner)
    assert cancelling.start_many((normal, local), "default", 2048)
    cancelling.wait()
    snapshot = cancelling.snapshot()
    assert call_count == 1
    assert snapshot.cancelled is True
    assert snapshot.stage == "Cancelled"
    assert snapshot.completed == 1

    broken = QueryController(
        config_path,
        runner=lambda *args: (_ for _ in ()).throw(RuntimeError("multi boom")),
    )
    assert broken.start_many((normal,), "default", 2048)
    broken.wait()
    snapshot = broken.snapshot()
    assert snapshot.stage == "Failed"
    assert snapshot.error == "multi boom"


def test_collect_documents_auto_scope_chooses_pdf_or_abstract(
    config_path, paper_record, monkeypatch
):
    config = load_config(config_path)
    no_pdf = replace(
        paper_record,
        openalex_id="W-AUTO-ABSTRACT",
        doi="10.1000/auto-abstract",
        arxiv_id=None,
        title="Auto abstract paper",
        pdf_url=None,
    )
    with Database(config.paths.database) as db:
        pdf_id, _ = db.upsert_paper(paper_record)
        abstract_id, _ = db.upsert_paper(no_pdf)

        def fake_paper_text(cfg, database, paper_id, scope, client):
            assert paper_id == pdf_id
            return database.get_paper(paper_id), "complete PDF text"

        monkeypatch.setattr("paperwatch.query_jobs._paper_text", fake_paper_text)
        documents = collect_query_documents(
            db,
            "auto",
            selection={"paper_ids": [pdf_id, abstract_id]},
            config=config,
            pdf_client=object(),
        )
    by_id = {item["paper_id"]: item for item in documents}
    assert by_id[pdf_id]["source"] == "fulltext"
    assert by_id[pdf_id]["fulltext"] == "complete PDF text"
    assert by_id[abstract_id]["source"] == "abstract"


def test_query_controller_scope_mismatch_and_invalid_snapshot_id(
    config_path, author_candidate
):
    entry_id, normal, _local, papers = _seed_multi_blog_queries(
        config_path, author_candidate
    )
    controller = QueryController(
        config_path, runner=lambda *args: {"completed": 1, "failed": 0}
    )
    with pytest.raises(ValueError, match="papers only"):
        controller.start(
            papers,
            ("default",),
            2048,
            selection={"blog_entry_ids": [entry_id]},
            content_type="blogs",
        )
    with pytest.raises(ValueError, match="No blog entries"):
        controller.start(
            normal,
            ("default",),
            2048,
            selection={"content_type": "blogs", "blog_entry_ids": ["invalid"]},
            content_type="blogs",
        )


def test_run_query_batch_cancels_all_pending_runs_before_model(
    config_path, author_candidate
):
    entry_id, normal, _local, _papers = _seed_multi_blog_queries(
        config_path, author_candidate
    )
    config = load_config(config_path)
    batch_id = "cancel-before-model"
    with Database(config.paths.database) as db:
        db.create_query_runs(
            batch_id=batch_id,
            query_id=normal,
            profiles=[
                ("default", "ollama", "test-model"),
                ("second", "ollama", "second-model"),
            ],
            context_window_tokens=2048,
            selection={
                "content_type": "blogs",
                "blog_entry_ids": [entry_id],
                "source_scope": "blog",
                "pdf_strategy": "auto",
            },
        )
    cancelled = threading.Event()
    cancelled.set()
    result = run_query_batch(
        config_path,
        batch_id,
        normal,
        ("default", "second"),
        2048,
        cancel_event=cancelled,
    )
    assert result == {"completed": 0, "failed": 0, "cancelled": True}
    with Database(config.paths.database) as db:
        rows = db.query_runs_for_batch(batch_id)
        assert all(row["status"] == "failed" for row in rows)
        assert all(row["error_text"] == "Cancelled by user" for row in rows)
