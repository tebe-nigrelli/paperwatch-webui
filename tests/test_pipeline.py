from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import date
from pathlib import Path

import httpx
import pytest

from paperwatch.db import Database
from paperwatch.models import BatchQueryResult, LLMResponse
from paperwatch.pipeline import PaperWatch, RunStats
from paperwatch.sources.arxiv import ArxivEntry


PDF_BYTES = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF\n"


class FakeOpenAlex:
    def __init__(self, papers=None, error=None):
        self.papers = papers or []
        self.error = error
        self.calls = []

    def recent_works(self, author_id, *, since, max_results):
        self.calls.append((author_id, since, max_results))
        if self.error:
            raise self.error
        return [deepcopy(paper) for paper in self.papers]

    def close(self):
        pass


class FakeArxiv:
    def __init__(self, entry):
        self.entry = entry
        self.by_id_calls = []
        self.by_title_calls = []

    def by_id(self, arxiv_id):
        self.by_id_calls.append(arxiv_id)
        return self.entry

    def by_title(self, title, first_author=None):
        self.by_title_calls.append((title, first_author))
        return self.entry

    def close(self):
        pass


class FakeLLM:
    provider = "fake"
    model = "fake-model"

    def __init__(self, result):
        self.result = result
        self.prompts = []

    def analyze(self, prompt):
        self.prompts.append(prompt)
        return LLMResponse(result=self.result, usage={"tokens": 1})

    def close(self):
        pass


def test_end_to_end_enrichment_pdf_analysis_and_cache(
    app_config,
    tmp_path,
    author_candidate,
    paper_record,
    analysis_result,
    monkeypatch,
):
    paper_record.abstract = None
    paper_record.abstract_source = None
    paper_record.pdf_url = None
    paper_record.pdf_source = None
    paper_record.arxiv_id = None

    entry = ArxivEntry(
        arxiv_id="2607.12345",
        title=paper_record.title,
        abstract="Fallback abstract from arXiv.",
        authors=["Ada Lovelace", "Alan Turing"],
        published="2026-07-18T00:00:00Z",
        updated="2026-07-20T00:00:00Z",
        abstract_url="https://arxiv.org/abs/2607.12345",
        pdf_url="https://arxiv.org/pdf/2607.12345.pdf",
        doi="10.1000/example",
    )
    openalex = FakeOpenAlex([paper_record])
    arxiv = FakeArxiv(entry)
    llm = FakeLLM(analysis_result)
    pdf_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=PDF_BYTES))
    )
    monkeypatch.setattr("paperwatch.pipeline.extract_pdf_text", lambda path, max_chars: "full paper text")

    with Database(app_config.paths.database) as db:
        db.add_author(author_candidate)
        app = PaperWatch(
            app_config,
            db=db,
            openalex=openalex,
            arxiv=arxiv,
            llm=llm,
            pdf_client=pdf_client,
        )
        first = app.run(
            today=date(2026, 7, 24), analyze_discovered_papers=True
        )
        second = app.run(
            today=date(2026, 7, 24), analyze_discovered_papers=True
        )

        assert first.status == "success"
        assert first.authors_checked == 1
        assert first.papers_found == 1
        assert first.papers_analyzed == 2
        assert first.pdfs_downloaded == 1
        assert second.papers_analyzed == 0
        assert second.pdfs_downloaded == 0
        assert len(llm.prompts) == 2
        assert openalex.calls[0][1] == "2026-06-24"
        stored = db.list_papers()[0]
        assert stored["abstract"] == "Fallback abstract from arXiv."
        assert stored["abstract_source"] == "arxiv"
        assert stored["local_pdf_path"]
        assert db._conn.execute("SELECT count(*) FROM analyses").fetchone()[0] == 2
        assert arxiv.by_title_calls


def test_pdf_is_always_processed_when_enabled(
    app_config, author_candidate, paper_record, analysis_result, monkeypatch
):
    llm = FakeLLM(analysis_result)
    openalex = FakeOpenAlex([paper_record])
    pdf_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=PDF_BYTES))
    )
    monkeypatch.setattr("paperwatch.pipeline.extract_pdf_text", lambda path, max_chars: "full text")
    with Database(app_config.paths.database) as db:
        db.add_author(author_candidate)
        app = PaperWatch(
            app_config,
            db=db,
            openalex=openalex,
            arxiv=None,
            llm=llm,
            pdf_client=pdf_client,
        )
        stats = app.run(
            today=date(2026, 7, 24), analyze_discovered_papers=True
        )
        assert stats.status == "success"
        assert stats.papers_analyzed == 2
        assert stats.pdfs_downloaded == 1
        assert db.list_papers()[0]["local_pdf_path"] is not None


def test_fetch_does_not_run_models_without_automatic_query_selection(
    app_config, author_candidate, paper_record
):
    class NoImplicitLLM:
        provider = "ollama"
        model = "test-model"

        def check_ready(self):
            raise AssertionError("fetch must not preflight a model")

        def analyze(self, prompt):
            raise AssertionError("fetch must not run an implicit summary")

        def close(self):
            pass

    with Database(app_config.paths.database) as db:
        db.add_author(author_candidate)
        app = PaperWatch(
            app_config,
            db=db,
            openalex=FakeOpenAlex([paper_record]),
            arxiv=None,
            llm=NoImplicitLLM(),
            pdf_client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, content=PDF_BYTES)
                )
            ),
        )
        stats = app.run(today=date(2026, 7, 24))
        assert stats.status == "success"
        assert stats.pdfs_downloaded == 1
        assert stats.papers_analyzed == 0
        assert db._conn.execute("SELECT count(*) FROM analyses").fetchone()[0] == 0
        assert db.list_query_runs() == []


def test_discovery_error_is_recorded_without_crashing(app_config, author_candidate, analysis_result):
    with Database(app_config.paths.database) as db:
        db.add_author(author_candidate)
        app = PaperWatch(
            app_config,
            db=db,
            openalex=FakeOpenAlex(error=RuntimeError("network down")),
            arxiv=None,
            llm=FakeLLM(analysis_result),
            pdf_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500))),
        )
        stats = app.run(
            today=date(2026, 7, 24), analyze_discovered_papers=True
        )
        assert stats.status == "partial"
        assert stats.authors_checked == 0
        assert stats.errors_count == 1
        assert db._conn.execute("SELECT stage FROM run_errors").fetchone()[0] == "paper_discovery"


def test_pdf_error_does_not_erase_abstract_analysis(
    app_config, author_candidate, paper_record, analysis_result
):
    llm = FakeLLM(analysis_result)
    pdf_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"not a pdf"))
    )
    with Database(app_config.paths.database) as db:
        db.add_author(author_candidate)
        app = PaperWatch(
            app_config,
            db=db,
            openalex=FakeOpenAlex([paper_record]),
            arxiv=None,
            llm=llm,
            pdf_client=pdf_client,
        )
        stats = app.run(
            today=date(2026, 7, 24), analyze_discovered_papers=True
        )
        assert stats.status == "partial"
        assert stats.papers_analyzed == 1
        assert stats.errors_count == 1
        assert db._conn.execute("SELECT count(*) FROM analyses").fetchone()[0] == 1
        assert db._conn.execute("SELECT stage FROM run_errors").fetchone()[0] == "pdf_processing"


def test_pdf_can_be_downloaded_without_llm(app_config, author_candidate, paper_record):
    config = replace(app_config, llm=replace(app_config.llm, provider="none", analyze_pdfs=False))
    pdf_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=PDF_BYTES))
    )

    class NoLLM:
        provider = "none"
        model = "none"

        def analyze(self, prompt):
            raise AssertionError("LLM should not be called")

        def close(self):
            pass

    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        app = PaperWatch(
            config,
            db=db,
            openalex=FakeOpenAlex([paper_record]),
            arxiv=None,
            llm=NoLLM(),
            pdf_client=pdf_client,
        )
        stats = app.run(
            today=date(2026, 7, 24), analyze_discovered_papers=True
        )
        assert stats.status == "success"
        assert stats.papers_analyzed == 0
        assert stats.pdfs_downloaded == 1


def test_complete_openalex_record_skips_arxiv_and_downloads_pdf_even_when_legacy_flag_is_false(
    app_config, author_candidate, paper_record, analysis_result
):
    class FailArxiv:
        def by_id(self, value):
            raise AssertionError("arXiv should not be queried")

        def by_title(self, *args):
            raise AssertionError("arXiv should not be queried")

    config = replace(
        app_config,
        fetch=replace(app_config.fetch, download_pdfs=False),
        llm=replace(app_config.llm, analyze_pdfs=False),
    )
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        app = PaperWatch(
            config,
            db=db,
            openalex=FakeOpenAlex([paper_record]),
            arxiv=FailArxiv(),
            llm=FakeLLM(analysis_result),
            pdf_client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, content=PDF_BYTES)
                )
            ),
        )
        stats = app.run(
            today=date(2026, 7, 24), analyze_discovered_papers=True
        )
        assert stats.status == "success"
        assert stats.pdfs_downloaded == 1
        assert db.list_papers()[0]["local_pdf_path"]


def test_arxiv_id_lookup_and_no_match_paths(app_config, author_candidate, paper_record, analysis_result):
    paper_record.abstract = None
    paper_record.pdf_url = None
    entry = ArxivEntry(
        arxiv_id=paper_record.arxiv_id,
        title=paper_record.title,
        abstract="From id",
        authors=["Ada Lovelace"],
        published=None,
        updated=None,
        abstract_url="https://arxiv.org/abs/2607.12345",
        pdf_url=None,
    )
    arxiv = FakeArxiv(entry)
    config = replace(app_config, fetch=replace(app_config.fetch, download_pdfs=False))
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        app = PaperWatch(
            config,
            db=db,
            openalex=FakeOpenAlex([paper_record]),
            arxiv=arxiv,
            llm=FakeLLM(analysis_result),
            pdf_client=httpx.Client(),
        )
        app.run(today=date(2026, 7, 24))
        assert arxiv.by_id_calls == ["2607.12345"]
        assert not arxiv.by_title_calls

    class NoMatch(FakeArxiv):
        def by_id(self, arxiv_id):
            self.by_id_calls.append(arxiv_id)
            return None

        def by_title(self, title, first_author=None):
            self.by_title_calls.append((title, first_author))
            return None

    paper_record.abstract = "Existing"
    paper_record.pdf_url = None
    no_match = NoMatch(None)
    with Database(config.paths.database.parent / "second.db") as db:
        db.add_author(author_candidate)
        app = PaperWatch(
            config,
            db=db,
            openalex=FakeOpenAlex([paper_record]),
            arxiv=no_match,
            llm=FakeLLM(analysis_result),
            pdf_client=httpx.Client(),
        )
        app.run(today=date(2026, 7, 24))
        assert no_match.by_title_calls


def test_duplicate_paper_across_authors_is_processed_once(
    app_config, author_candidate, paper_record, analysis_result
):
    from paperwatch.models import AuthorCandidate

    config = replace(app_config, fetch=replace(app_config.fetch, download_pdfs=False))
    openalex = FakeOpenAlex([paper_record])
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        db.add_author(AuthorCandidate(openalex_id="A999", name="Alan Turing"))
        app = PaperWatch(
            config,
            db=db,
            openalex=openalex,
            arxiv=None,
            llm=FakeLLM(analysis_result),
            pdf_client=httpx.Client(),
        )
        stats = app.run(
            today=date(2026, 7, 24), analyze_discovered_papers=True
        )
        assert stats.authors_checked == 2
        assert stats.papers_found == 1
        assert stats.papers_analyzed == 1
        assert len(openalex.calls) == 2


def test_no_abstract_no_pdf_and_pdf_disabled_paths(
    app_config, author_candidate, paper_record, analysis_result
):
    paper_record.abstract = None
    paper_record.pdf_url = None
    config = replace(
        app_config,
        arxiv=replace(app_config.arxiv, enabled=False),
        fetch=replace(app_config.fetch, download_pdfs=False),
    )
    llm = FakeLLM(analysis_result)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        app = PaperWatch(
            config,
            db=db,
            openalex=FakeOpenAlex([paper_record]),
            arxiv=None,
            llm=llm,
            pdf_client=httpx.Client(),
        )
        stats = app.run(
            today=date(2026, 7, 24), analyze_discovered_papers=True
        )
        assert stats.papers_analyzed == 0
        assert llm.prompts == []


def test_pdf_download_without_fulltext_analysis(
    app_config, author_candidate, paper_record, analysis_result
):
    config = replace(app_config, llm=replace(app_config.llm, analyze_pdfs=False))
    pdf_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=PDF_BYTES))
    )
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        app = PaperWatch(
            config,
            db=db,
            openalex=FakeOpenAlex([paper_record]),
            arxiv=None,
            llm=FakeLLM(analysis_result),
            pdf_client=pdf_client,
        )
        stats = app.run(
            today=date(2026, 7, 24), analyze_discovered_papers=True
        )
        assert stats.pdfs_downloaded == 1
        assert stats.papers_analyzed == 1


def test_arxiv_and_abstract_errors_are_isolated(
    app_config, author_candidate, paper_record, analysis_result
):
    paper_record.pdf_url = None

    class BrokenArxiv:
        def by_id(self, value):
            raise RuntimeError("arxiv down")

    class BrokenLLM(FakeLLM):
        def analyze(self, prompt):
            raise RuntimeError("llm down")

    config = replace(app_config, fetch=replace(app_config.fetch, download_pdfs=False))
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        app = PaperWatch(
            config,
            db=db,
            openalex=FakeOpenAlex([paper_record]),
            arxiv=BrokenArxiv(),
            llm=BrokenLLM(analysis_result),
            pdf_client=httpx.Client(),
        )
        stats = app.run(
            today=date(2026, 7, 24), analyze_discovered_papers=True
        )
        assert stats.status == "partial"
        assert stats.errors_count == 2
        stages = {row[0] for row in db._conn.execute("SELECT stage FROM run_errors")}
        assert stages == {"arxiv_enrichment", "abstract_analysis"}


def test_paper_processing_and_fatal_run_errors(app_config, author_candidate, paper_record, analysis_result, monkeypatch):
    config = replace(app_config, fetch=replace(app_config.fetch, download_pdfs=False))
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        monkeypatch.setattr(db, "upsert_paper", lambda paper: (_ for _ in ()).throw(RuntimeError("db write")))
        app = PaperWatch(
            config,
            db=db,
            openalex=FakeOpenAlex([paper_record]),
            arxiv=None,
            llm=FakeLLM(analysis_result),
            pdf_client=httpx.Client(),
        )
        stats = app.run(
            today=date(2026, 7, 24), analyze_discovered_papers=True
        )
        assert stats.status == "partial"
        assert db._conn.execute("SELECT stage FROM run_errors").fetchone()[0] == "paper_processing"

    with Database(config.paths.database.parent / "fatal.db") as db:
        monkeypatch.setattr(db, "list_authors", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fatal")))
        app = PaperWatch(
            config,
            db=db,
            openalex=FakeOpenAlex([]),
            arxiv=None,
            llm=FakeLLM(analysis_result),
            pdf_client=httpx.Client(),
        )
        import pytest

        with pytest.raises(RuntimeError, match="fatal"):
            app.run(today=date(2026, 7, 24))
        assert db.list_runs()[0]["status"] == "failed"


def test_metadata_only_does_not_require_prompt_and_context_closes_client(
    app_config, author_candidate
):
    config = replace(
        app_config,
        paths=replace(app_config.paths, prompt=app_config.paths.prompt.parent / "missing.txt"),
        llm=replace(app_config.llm, provider="none"),
    )

    class NoLLM:
        provider = "none"
        model = "none"

        def analyze(self, prompt):
            raise AssertionError

        def close(self):
            pass

    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        app = PaperWatch(config, db=db, openalex=FakeOpenAlex([]), arxiv=None, llm=NoLLM())
        client = app.pdf_client
        with app:
            assert app.run(today=date(2026, 7, 24)).status == "success"
        assert client.is_closed


def test_pipeline_enforces_exact_title_and_configured_model(
    app_config, author_candidate, paper_record, analysis_result
):
    from dataclasses import replace

    config = replace(
        app_config,
        fetch=replace(app_config.fetch, download_pdfs=False),
        llm=replace(app_config.llm, analyze_pdfs=False),
    )
    wrong = analysis_result.model_copy(
        update={"name": "Hallucinated title", "generated_by_model": "hallucinated-model"}
    )
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        app = PaperWatch(
            config,
            db=db,
            openalex=FakeOpenAlex([paper_record]),
            arxiv=None,
            llm=FakeLLM(wrong),
            pdf_client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, content=PDF_BYTES)
                )
            ),
        )
        stats = app.run(
            today=date(2026, 7, 24), analyze_discovered_papers=True
        )
        assert stats.status == "success"
        row = db._conn.execute("SELECT name, generated_by_model FROM analyses").fetchone()
        assert row["name"] == paper_record.title
        assert row["generated_by_model"] == "fake-model (version not exposed)"


def test_daily_fulltext_automatically_uses_chunking_when_direct_prompt_does_not_fit(
    app_config, paper_record, analysis_result, monkeypatch
):
    from paperwatch.analysis import AnalysisOutcome
    from paperwatch.chunking import ChunkedAnalysisOutcome
    from paperwatch.pipeline import RunStats

    llm = FakeLLM(analysis_result)
    calls = []
    monkeypatch.setattr(
        "paperwatch.pipeline.direct_prompt_fit", lambda **kwargs: (False, 20000, 14000)
    )

    def chunked(**kwargs):
        calls.append(kwargs["text"])
        analysis_id = kwargs["db"].save_analysis(
            paper_id=kwargs["paper_id"],
            scope="fulltext",
            provider="fake",
            model="fake-model",
            prompt_version=app_config.llm.prompt_version,
            content_sha256="daily-chunked",
            input_chars=len(kwargs["text"]),
            result=analysis_result,
            usage={},
            analysis_strategy="chunked",
            context_window_tokens=app_config.llm.context_window_tokens,
            chunk_count=2,
        )
        return ChunkedAnalysisOutcome(
            AnalysisOutcome(analysis_result, analysis_id, True), 2
        )

    monkeypatch.setattr("paperwatch.pipeline.analyze_chunked_text", chunked)
    with Database(app_config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        app = PaperWatch(
            app_config,
            db=db,
            openalex=FakeOpenAlex(),
            arxiv=None,
            llm=llm,
            pdf_client=httpx.Client(),
        )
        stats = RunStats()
        result = app._analyze(
            stats,
            paper_id=paper_id,
            paper=paper_record,
            text="complete full text",
            scope="fulltext",
        )
        assert result == analysis_result
        assert calls == ["complete full text"]
        assert stats.papers_analyzed == 1


def test_missing_ollama_model_is_reported_once_and_preserves_initial_cursor(
    app_config, author_candidate, paper_record, analysis_result
):
    class MissingModelLLM(FakeLLM):
        provider = "ollama"

        def check_ready(self):
            raise RuntimeError(
                "Ollama model 'qwen3:8b' is not installed. Run `ollama pull qwen3:8b`."
            )

        def analyze(self, prompt):
            raise AssertionError("analysis must be skipped after failed preflight")

    config = replace(app_config, fetch=replace(app_config.fetch, download_pdfs=False))
    llm = MissingModelLLM(analysis_result)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        app = PaperWatch(
            config,
            db=db,
            openalex=FakeOpenAlex([paper_record]),
            arxiv=None,
            llm=llm,
            pdf_client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, content=PDF_BYTES)
                )
            ),
        )
        stats = app.run(
            today=date(2026, 7, 24), analyze_discovered_papers=True
        )
        assert stats.status == "partial"
        assert stats.errors_count == 1
        assert stats.pdfs_downloaded == 1
        error = db._conn.execute(
            "SELECT stage, message FROM run_errors"
        ).fetchone()
        assert error["stage"] == "llm_preflight"
        assert "ollama pull qwen3:8b" in error["message"]
        author = db.get_author(author_candidate.openalex_id)
        assert author["last_checked_at"] is None
        assert db._conn.execute("SELECT count(*) FROM papers").fetchone()[0] == 1
        assert db._conn.execute("SELECT count(*) FROM analyses").fetchone()[0] == 0


def test_daily_codex_fulltext_uploads_original_pdf_without_text_extraction(
    app_config, author_candidate, paper_record, analysis_result, monkeypatch
):
    config = replace(
        app_config,
        llm=replace(
            app_config.llm,
            provider="codex",
            model="gpt-5.3-codex",
            api_key="secret",
            reasoning_effort="high",
        ),
    )

    class FakeCodex(FakeLLM):
        provider = "codex"
        model = "gpt-5.3-codex"

        def __init__(self, result):
            super().__init__(result)
            self.uploads = []

        def check_ready(self):
            pass

        def analyze_pdf(self, prompt, pdf_path):
            self.uploads.append((prompt, Path(pdf_path).read_bytes()))
            return LLMResponse(result=self.result, usage={"input_tokens": 100})

    from pathlib import Path

    llm = FakeCodex(analysis_result)
    pdf_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=PDF_BYTES))
    )
    monkeypatch.setattr(
        "paperwatch.pipeline.extract_pdf_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not extract")),
    )
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        app = PaperWatch(
            config,
            db=db,
            openalex=FakeOpenAlex([paper_record]),
            arxiv=None,
            llm=llm,
            pdf_client=pdf_client,
        )
        stats = app.run(
            today=date(2026, 7, 24), analyze_discovered_papers=True
        )
        assert stats.status == "success"
        assert stats.papers_analyzed == 2
        assert llm.uploads[0][1] == PDF_BYTES
        analyses = db.paper_analyses(db.list_papers()[0]["id"])
        assert {row["analysis_strategy"] for row in analyses} == {"direct", "pdf_upload"}


def test_daily_claude_code_reads_original_pdf_without_text_extraction(
    app_config, author_candidate, paper_record, analysis_result, monkeypatch
):
    config = replace(
        app_config,
        llm=replace(
            app_config.llm,
            provider="claude-code",
            model="claude-sonnet-4-5",
            api_key=None,
            reasoning_effort="high",
            pdf_analysis_strategy="chunked",
        ),
    )

    class FakeClaude(FakeLLM):
        provider = "claude-code"
        model = "claude-sonnet-4-5"

        def __init__(self, result):
            super().__init__(result)
            self.uploads = []

        def analyze_pdf(self, prompt, pdf_path):
            self.uploads.append((prompt, Path(pdf_path).read_bytes()))
            return LLMResponse(result=self.result, usage={"input_tokens": 100})

    llm = FakeClaude(analysis_result)
    pdf_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=PDF_BYTES))
    )
    monkeypatch.setattr(
        "paperwatch.pipeline.extract_pdf_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not extract")),
    )
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        app = PaperWatch(
            config,
            db=db,
            openalex=FakeOpenAlex([paper_record]),
            arxiv=None,
            llm=llm,
            pdf_client=pdf_client,
        )
        stats = app.run(
            today=date(2026, 7, 24), analyze_discovered_papers=True
        )
        assert stats.status == "success"
        assert stats.papers_analyzed == 2
        assert llm.uploads[0][1] == PDF_BYTES
        analyses = db.paper_analyses(db.list_papers()[0]["id"])
        assert {row["analysis_strategy"] for row in analyses} == {"direct", "pdf_upload"}


def test_cursor_independent_download_only_backfill_and_late_pdf_analysis(
    app_config,
    author_candidate,
    paper_record,
    analysis_result,
    monkeypatch,
):
    openalex = FakeOpenAlex([paper_record])
    llm = FakeLLM(analysis_result)
    pdf_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=PDF_BYTES))
    )
    monkeypatch.setattr(
        "paperwatch.pipeline.extract_pdf_text", lambda path, max_chars: "complete paper text"
    )

    with Database(app_config.paths.database) as db:
        db.add_author(author_candidate)
        db.mark_author_checked("A123", "2026-07-29T00:00:00+00:00")
        app = PaperWatch(
            app_config,
            db=db,
            openalex=openalex,
            arxiv=None,
            llm=llm,
            pdf_client=pdf_client,
        )
        backfill = app.run(
            today=date(2026, 7, 30),
            since_override=date(2025, 7, 30),
            download_only=True,
            historical=True,
            max_papers_per_author_override=5000,
        )
        assert backfill.status == "success"
        assert backfill.pdfs_downloaded == 1
        assert backfill.papers_analyzed == 0
        assert openalex.calls[0] == ("A123", "2025-07-30", 5000)
        stored = db.list_papers()[0]
        assert stored["auto_analyze"] == 0
        assert stored["local_pdf_path"]
        assert llm.prompts == []
        assert db.get_author("A123")["last_checked_at"] > "2026-07-29T00:00:00+00:00"

        # Rediscovery does not summarize an already downloaded historical paper.
        normal = app.run(
            today=date(2026, 7, 31), analyze_discovered_papers=True
        )
        assert normal.papers_analyzed == 0
        assert llm.prompts == []

        # If the PDF first becomes downloadable later, analyze that new download once.
        Path(stored["local_pdf_path"]).unlink()
        late = app.run(
            today=date(2026, 8, 1), analyze_discovered_papers=True
        )
        assert late.pdfs_downloaded == 1
        assert late.papers_analyzed == 1
        assert len(llm.prompts) == 1
        assert db.paper_analyses(stored["id"])[0]["scope"] == "fulltext"


def test_backfill_override_validation_happens_before_run_creation(
    app_config, author_candidate, analysis_result
):
    import pytest

    with Database(app_config.paths.database) as db:
        db.add_author(author_candidate)
        app = PaperWatch(
            app_config,
            db=db,
            openalex=FakeOpenAlex([]),
            arxiv=None,
            llm=FakeLLM(analysis_result),
            pdf_client=httpx.Client(),
        )
        with pytest.raises(ValueError, match="at least 1"):
            app.run(max_papers_per_author_override=0)
        assert db.list_runs() == []


def test_fetch_title_selects_exact_match_and_downloads(
    app_config, paper_record, analysis_result
):
    from dataclasses import replace

    near = replace(paper_record, openalex_id="WNEAR", title="Minimal Agent Systems")

    class SearchOpenAlex:
        def __init__(self):
            self.calls = []

        def search_works(self, title, limit=10):
            self.calls.append((title, limit))
            return [near, paper_record]

    openalex = SearchOpenAlex()
    pdf_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=PDF_BYTES)
        )
    )
    with Database(app_config.paths.database) as db:
        app = PaperWatch(
            app_config,
            db=db,
            openalex=openalex,
            arxiv=None,
            llm=FakeLLM(analysis_result),
            pdf_client=pdf_client,
        )
        stats, paper_id = app.fetch_title(paper_record.title)
        assert stats.status == "success"
        assert stats.papers_found == 1 and stats.pdfs_downloaded == 1
        assert openalex.calls == [(paper_record.title, 10)]
        stored = db.get_paper(paper_id)
        assert stored["openalex_id"] == paper_record.openalex_id
        assert Path(stored["local_pdf_path"]).is_file()
        assert db.paper_analyses(paper_id) == []
        assert app._best_title_match("Minimal Agent System", [near]) == near
        with pytest.raises(ValueError, match="cannot be empty"):
            app.fetch_title("   ")


def test_fetch_title_records_missing_match_and_pdf_errors(
    app_config, paper_record, analysis_result
):
    from dataclasses import replace

    class SearchOpenAlex:
        def __init__(self, papers):
            self.papers = papers

        def search_works(self, title, limit=10):
            return self.papers

    with Database(app_config.paths.database) as db:
        app = PaperWatch(
            app_config,
            db=db,
            openalex=SearchOpenAlex([]),
            arxiv=None,
            llm=FakeLLM(analysis_result),
            pdf_client=httpx.Client(),
        )
        stats, paper_id = app.fetch_title("Unknown paper")
        assert stats.status == "failed" and paper_id is None
        assert db.get_run_errors(stats.run_id)[0]["stage"] == "paper_search_download"

        no_pdf = replace(
            paper_record,
            openalex_id="WNOFILE",
            doi=None,
            arxiv_id=None,
            title="Paper without any PDF source",
            publication_url="https://publisher.test/no-file",
            pdf_url=None,
        )
        app.openalex = SearchOpenAlex([no_pdf])
        stats, paper_id = app.fetch_title(no_pdf.title)
        assert stats.status == "failed" and paper_id is not None
        assert "no downloadable PDF" in db.get_run(stats.run_id)["error_text"]


def test_fetch_title_keeps_download_when_arxiv_enrichment_fails(
    app_config, paper_record, analysis_result
):
    from dataclasses import replace

    paper = replace(paper_record, abstract=None, arxiv_id=None)

    class SearchOpenAlex:
        def search_works(self, title, limit=10):
            return [paper]

    class BrokenArxiv:
        def by_title(self, title, first_author=None):
            raise RuntimeError("arXiv offline")

        def by_id(self, arxiv_id):
            raise AssertionError("no arXiv id expected")

    pdf_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=PDF_BYTES)
        )
    )
    with Database(app_config.paths.database) as db:
        app = PaperWatch(
            app_config,
            db=db,
            openalex=SearchOpenAlex(),
            arxiv=BrokenArxiv(),
            llm=FakeLLM(analysis_result),
            pdf_client=pdf_client,
        )
        stats, paper_id = app.fetch_title(paper.title)
        assert stats.status == "partial"
        assert stats.pdfs_downloaded == 1 and paper_id is not None
        errors = db.get_run_errors(stats.run_id)
        assert errors[0]["stage"] == "arxiv_enrichment"


def test_fetch_work_downloads_exact_result_and_keeps_failed_record(
    app_config, paper_record, analysis_result
):
    from dataclasses import replace

    class WorkOpenAlex:
        def __init__(self, paper=None, error=None):
            self.paper = paper
            self.error = error
            self.calls = []

        def get_work(self, work_id):
            self.calls.append(work_id)
            if self.error:
                raise self.error
            return self.paper

    pdf_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=PDF_BYTES)
        )
    )
    source = WorkOpenAlex(paper_record)
    with Database(app_config.paths.database) as db:
        app = PaperWatch(
            app_config,
            db=db,
            openalex=source,
            arxiv=None,
            llm=FakeLLM(analysis_result),
            pdf_client=pdf_client,
        )
        stats, paper_id = app.fetch_work("W456")
        assert stats.status == "success" and paper_id is not None
        assert source.calls == ["W456"]
        assert Path(db.get_paper(paper_id)["local_pdf_path"]).is_file()
        with pytest.raises(ValueError, match="cannot be empty"):
            app.fetch_work(" ")

        no_pdf = replace(
            paper_record,
            openalex_id="WNOFILE",
            doi=None,
            arxiv_id=None,
            title="Paper without any PDF source",
            publication_url="https://publisher.test/no-file",
            pdf_url=None,
        )
        app.openalex = WorkOpenAlex(no_pdf)
        app.pdf_client = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    content=b"<html><body>No PDF</body></html>",
                    headers={"content-type": "text/html"},
                )
            )
        )
        failed, failed_paper_id = app.fetch_work("WNOFILE")
        assert failed.status == "failed" and failed_paper_id is not None
        assert db.get_run_errors(failed.run_id)[0]["stage"] == "paper_download"

        app.openalex = WorkOpenAlex(error=RuntimeError("OpenAlex unavailable"))
        failed, failed_paper_id = app.fetch_work("WBROKEN")
        assert failed.status == "failed" and failed_paper_id is None
        assert "OpenAlex unavailable" in db.get_run(failed.run_id)["error_text"]


def test_manual_fetch_downloads_pdf_and_checks_blogs(
    app_config, author_candidate, paper_record, monkeypatch
):
    from paperwatch.blogs import BlogFetch

    config = replace(
        app_config,
        fetch=replace(app_config.fetch, download_pdfs=False),
        llm=replace(app_config.llm, provider="none", analyze_pdfs=False),
    )
    fetched_urls = []

    def fake_blog(client, row):
        fetched_urls.append(row["url"])
        return BlogFetch(
            status_code=200,
            title="Blog",
            method="snapshot",
            snapshot_text="baseline text",
            snapshot_sha256="baseline-hash",
        )

    monkeypatch.setattr("paperwatch.pipeline.fetch_blog", fake_blog)

    class WorkOpenAlex(FakeOpenAlex):
        def get_work(self, work_id):
            return paper_record

    class NoLLM:
        provider = "none"
        model = "none"

        def close(self):
            pass

    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        db.add_author_blog(author_candidate.openalex_id, "https://example.test/blog")
        app = PaperWatch(
            config,
            db=db,
            openalex=WorkOpenAlex(),
            arxiv=None,
            llm=NoLLM(),
            pdf_client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, content=PDF_BYTES)
                )
            ),
        )
        stats, paper_id = app.fetch_work("W456")
        assert paper_id is not None
        assert stats.pdfs_downloaded == 1
        assert stats.blogs_checked == 1
        assert fetched_urls == ["https://example.test/blog"]
        assert db.get_paper(paper_id)["local_pdf_path"]

def test_automatic_saved_queries_target_new_papers_and_blogs(
    app_config, author_candidate, paper_record, analysis_result, monkeypatch
):
    class NoopOpenAlex:
        def close(self):
            pass

    captured: list[tuple[str, dict]] = []

    def collect(db, source_scope, **kwargs):
        selection = kwargs["selection"]
        captured.append((source_scope, selection))
        identity_key = "paper_ids" if selection["content_type"] == "papers" else "blog_entry_ids"
        return [
            {
                "title": "new record",
                identity_key[:-1]: selection[identity_key][0],
                "content": "evidence",
            }
        ]

    def run_model(llm, **kwargs):
        content_type = kwargs["content_type"]
        return (
            BatchQueryResult(
                title=f"{content_type} result",
                summary="done",
                items=[],
            ),
            {"tokens": 1},
            12,
        )

    monkeypatch.setattr("paperwatch.query_jobs.collect_query_documents", collect)
    monkeypatch.setattr("paperwatch.query_jobs.run_query_for_model", run_model)

    with Database(app_config.paths.database) as db:
        db.add_author(author_candidate)
        blog_id = db.add_author_blog(author_candidate.openalex_id, "https://example.test/blog")
        blog_entry_id = db.save_blog_entry(
            blog_id=blog_id,
            title="New blog entry",
            content_sha256="new-blog-entry",
            normalized_text="New blog evidence",
            diff_text="",
            change_kind="changed",
            previous_entry_id=None,
            entry_kind="rss",
            external_id="new-blog-entry",
            source_url="https://example.test/blog/new",
        )
        paper_id, _ = db.upsert_paper(paper_record)
        db.create_saved_query(
            name="New papers",
            prompt="Summarize new papers",
            auto_run_papers=True,
        )
        db.create_saved_query(
            name="New blogs",
            prompt="Summarize new blogs",
            auto_run_blogs=True,
        )
        app = PaperWatch(
            app_config,
            db=db,
            openalex=NoopOpenAlex(),
            arxiv=None,
            llm=FakeLLM(analysis_result),
            pdf_client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, content=PDF_BYTES)
                )
            ),
        )
        stats = RunStats(
            run_id=db.start_run(),
            new_paper_ids=[paper_id, paper_id],
            new_blog_entry_ids=[blog_entry_id, blog_entry_id],
        )
        app._run_automatic_queries(stats)
        runs = db.list_query_runs()
        assert len(runs) == 2
        assert {row["status"] for row in runs} == {"success"}
        assert {row["matched_count"] for row in runs} == {1}
        assert [scope for scope, _selection in captured] == ["auto", "blog"]
        paper_selection = captured[0][1]
        blog_selection = captured[1][1]
        assert paper_selection["paper_ids"] == [paper_id]
        assert blog_selection["blog_entry_ids"] == [blog_entry_id]
        assert paper_selection["content_type"] == "papers"
        assert blog_selection["content_type"] == "blogs"

        def fail_model(llm, **kwargs):
            raise RuntimeError("query failed")

        monkeypatch.setattr("paperwatch.query_jobs.run_query_for_model", fail_model)
        failed = RunStats(run_id=db.start_run(), new_blog_entry_ids=[blog_entry_id])
        app._run_automatic_queries(failed)
        assert failed.errors_count == 1
        assert db.get_run_errors(failed.run_id)[0]["stage"] == "automatic_blogs_query"
        assert db.list_query_runs()[0]["status"] == "failed"

        no_targets = RunStats(run_id=db.start_run())
        app._run_automatic_queries(no_targets)
        app._llm_available = False
        app._run_automatic_queries(
            RunStats(run_id=db.start_run(), new_paper_ids=[paper_id])
        )

def test_selected_fetch_checks_blogs_and_automatic_queries(
    app_config, author_candidate, paper_record, analysis_result, monkeypatch
):
    from paperwatch.blogs import BlogFetch

    class WorkOpenAlex:
        def get_work(self, work_id):
            return paper_record

    with Database(app_config.paths.database) as db:
        db.add_author(author_candidate)
        db.add_author_blog(author_candidate.openalex_id, "https://example.test/blog")
        app = PaperWatch(
            app_config,
            db=db,
            openalex=WorkOpenAlex(),
            arxiv=None,
            llm=FakeLLM(analysis_result),
            pdf_client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, content=PDF_BYTES)
                )
            ),
        )
        calls: list[str] = []

        def blogs(rows):
            calls.append("blogs")
            return [(
                rows[0],
                BlogFetch(
                    status_code=200,
                    title="Blog",
                    method="snapshot",
                    snapshot_text="baseline",
                    snapshot_sha256="baseline-hash",
                ),
                None,
            )]

        def queries(stats):
            calls.append("queries")

        monkeypatch.setattr(app, "_fetch_blog_batch", blogs)
        monkeypatch.setattr(app, "_run_automatic_queries", queries)
        stats, paper_id = app.fetch_work("W456")
        assert paper_id is not None
        assert stats.pdfs_downloaded == 1
        assert stats.blogs_checked == 1
        assert calls == ["blogs", "queries"]


def test_normal_fetch_starts_blog_network_work_while_papers_are_discovered(
    app_config, author_candidate, monkeypatch
):
    from paperwatch.blogs import BlogFetch
    import threading

    blog_started = threading.Event()
    paper_started = threading.Event()

    class ParallelOpenAlex:
        def recent_works(self, *args, **kwargs):
            assert blog_started.wait(1), "blog fetch did not start before paper discovery"
            paper_started.set()
            return []

    class NoLLM:
        provider = "none"
        model = "none"

        def close(self):
            pass

    def fake_blog(client, row):
        blog_started.set()
        assert paper_started.wait(1), "paper discovery did not overlap the blog fetch"
        return BlogFetch(
            status_code=200,
            title="Blog",
            method="snapshot",
            snapshot_text="baseline",
            snapshot_sha256="baseline-hash",
        )

    monkeypatch.setattr("paperwatch.pipeline.fetch_blog", fake_blog)
    with Database(app_config.paths.database) as db:
        db.add_author(author_candidate)
        db.add_author_blog(author_candidate.openalex_id, "https://example.test/blog")
        app = PaperWatch(
            app_config,
            db=db,
            openalex=ParallelOpenAlex(),
            arxiv=None,
            llm=NoLLM(),
            pdf_client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, text="unused")
                )
            ),
        )
        stats = app.run(today=date(2026, 8, 1))
        assert stats.status == "success"
        assert stats.authors_checked == 1
        assert stats.blogs_checked == 1



def test_available_pdf_is_archived_when_legacy_download_toggle_is_false_and_abstracts_are_disabled(
    app_config, author_candidate, paper_record, analysis_result, monkeypatch
):
    from paperwatch.analysis import AnalysisOutcome
    from paperwatch.chunking import ChunkedAnalysisOutcome

    config = replace(
        app_config,
        fetch=replace(app_config.fetch, download_pdfs=False),
        llm=replace(
            app_config.llm,
            analyze_abstracts=False,
            pdf_analysis_strategy="chunked",
        ),
    )

    class ChunkCapableLLM(FakeLLM):
        def generate_json(self, *args, **kwargs):
            raise AssertionError("chunk helper is patched")

    chunk_calls: list[str] = []

    def chunked(**kwargs):
        chunk_calls.append(kwargs["text"])
        analysis_id = kwargs["db"].save_analysis(
            paper_id=kwargs["paper_id"],
            scope="fulltext",
            provider="fake",
            model="fake-model",
            prompt_version=config.llm.prompt_version,
            content_sha256="default-chunked",
            input_chars=len(kwargs["text"]),
            result=analysis_result,
            usage={},
            analysis_strategy="chunked",
            context_window_tokens=config.llm.context_window_tokens,
            chunk_count=2,
        )
        return ChunkedAnalysisOutcome(
            AnalysisOutcome(analysis_result, analysis_id, True), 2
        )

    monkeypatch.setattr("paperwatch.pipeline.extract_pdf_text", lambda *args, **kwargs: "pdf text")
    monkeypatch.setattr("paperwatch.pipeline.analyze_chunked_text", chunked)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        app = PaperWatch(
            config,
            db=db,
            openalex=FakeOpenAlex([paper_record]),
            arxiv=None,
            llm=ChunkCapableLLM(analysis_result),
            pdf_client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, content=PDF_BYTES)
                )
            ),
        )
        stats = app.run(
            today=date(2026, 7, 24), analyze_discovered_papers=True
        )
        stored = db.list_papers()[0]
        assert stats.status == "success"
        assert stats.pdfs_downloaded == 1
        assert Path(stored["local_pdf_path"]).read_bytes() == PDF_BYTES
        assert chunk_calls == ["pdf text"]
        assert db._conn.execute(
            "SELECT count(*) FROM analyses WHERE scope='abstract'"
        ).fetchone()[0] == 0


def test_recent_missing_pdf_is_refreshed_and_retried(
    app_config, paper_record, analysis_result
):
    stored_paper = deepcopy(paper_record)
    stored_paper.pdf_url = None
    stored_paper.pdf_source = None

    class RefreshingOpenAlex(FakeOpenAlex):
        def __init__(self):
            super().__init__([])
            self.work_calls: list[str] = []

        def get_work(self, work_id):
            self.work_calls.append(work_id)
            return deepcopy(paper_record)

    class NoLLM:
        provider = "none"
        model = ""

        def close(self):
            pass

    config = replace(app_config, llm=replace(app_config.llm, provider="none", model=""))
    openalex = RefreshingOpenAlex()
    progress: list[tuple[str, str]] = []
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(stored_paper)
        app = PaperWatch(
            config,
            db=db,
            openalex=openalex,
            arxiv=None,
            llm=NoLLM(),
            progress=lambda kind, value: progress.append((kind, value)),
            pdf_client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, content=PDF_BYTES)
                )
            ),
        )
        stats = app.run(today=date(2026, 7, 24))
        row = db.get_paper(paper_id)
        assert stats.status == "success"
        assert stats.pdfs_downloaded == 1
        assert openalex.work_calls == ["W456"]
        assert Path(row["local_pdf_path"]).is_file()
        assert ("progress", "1/1") in progress
        assert any(value.startswith("Retried PDF") for kind, value in progress if kind == "message")


def test_fetch_run_can_be_cancelled_before_network_work(
    app_config, author_candidate, analysis_result
):
    import threading

    event = threading.Event()
    event.set()
    progress: list[tuple[str, str]] = []
    with Database(app_config.paths.database) as db:
        db.add_author(author_candidate)
        app = PaperWatch(
            app_config,
            db=db,
            openalex=FakeOpenAlex([]),
            arxiv=None,
            llm=FakeLLM(analysis_result),
            cancel_event=event,
            progress=lambda kind, value: progress.append((kind, value)),
        )
        stats = app.run(today=date(2026, 7, 24))
        assert stats.status == "cancelled"
        assert db.list_runs(1)[0]["status"] == "cancelled"
        assert ("progress", "0/1") in progress


def test_pdf_retry_uses_stored_url_when_refresh_fails_and_records_bad_pdf(
    app_config, paper_record
):
    first = deepcopy(paper_record)
    first.openalex_id = "WRETRY1"
    first.doi = "10.1000/retry1"
    first.arxiv_id = "2607.10001"
    first.title = "Retry from stored URL"
    first.pdf_url = "https://papers.test/good.pdf"
    first.publication_date = "2026-07-20"
    second = deepcopy(paper_record)
    second.openalex_id = "WRETRY2"
    second.doi = "10.1000/retry2"
    second.arxiv_id = "2607.10002"
    second.title = "Retry invalid PDF"
    second.pdf_url = "https://papers.test/bad.pdf"
    second.publication_date = "2026-07-19"

    class FailingRefresh(FakeOpenAlex):
        def get_work(self, work_id):
            raise RuntimeError("OpenAlex temporarily unavailable")

    class NoLLM:
        provider = "none"
        model = ""

        def close(self):
            pass

    def handler(request):
        if request.url.path.endswith("good.pdf"):
            return httpx.Response(200, content=PDF_BYTES)
        return httpx.Response(200, content=b"not a pdf")

    config = replace(app_config, llm=replace(app_config.llm, provider="none", model=""))
    with Database(config.paths.database) as db:
        first_id, _ = db.upsert_paper(first)
        db.upsert_paper(second)
        app = PaperWatch(
            config,
            db=db,
            openalex=FailingRefresh([]),
            arxiv=None,
            llm=NoLLM(),
            pdf_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        stats = app.run(today=date(2026, 7, 24))
        assert stats.status == "partial"
        assert stats.pdfs_downloaded == 1
        assert Path(db.get_paper(first_id)["local_pdf_path"]).is_file()
        errors = db.get_run_errors(stats.run_id)
        assert any(row["stage"] == "pdf_retry" and "Retry invalid PDF" in row["message"] for row in errors)


def test_missing_pdf_is_resolved_from_publication_page_and_stored(
    app_config, author_candidate, paper_record
):
    paper = deepcopy(paper_record)
    paper.pdf_url = None
    paper.pdf_source = None
    paper.arxiv_id = None
    paper.publication_url = "https://dl.acm.org/doi/10.1145/example"
    paper.metadata = {}

    class NoLLM:
        provider = "none"
        model = ""

        def close(self):
            pass

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/doi/pdf/10.1145/example":
            return httpx.Response(200, content=PDF_BYTES)
        return httpx.Response(200, text="<html>No PDF</html>", headers={"content-type": "text/html"})

    config = replace(app_config, llm=replace(app_config.llm, provider="none", model=""))
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        app = PaperWatch(
            config,
            db=db,
            openalex=FakeOpenAlex([paper]),
            arxiv=None,
            llm=NoLLM(),
            pdf_client=httpx.Client(
                transport=httpx.MockTransport(handler), follow_redirects=True
            ),
        )
        stats = app.run(today=date(2026, 7, 24))
        stored = db.list_papers()[0]
        assert stats.status == "success"
        assert stats.pdfs_downloaded == 1
        assert stored["pdf_url"] == "https://dl.acm.org/doi/pdf/10.1145/example"
        assert stored["pdf_source"] == "source_page"
        assert Path(stored["local_pdf_path"]).is_file()


def test_invalid_source_pdf_landing_page_falls_back_to_embedded_pdf(
    app_config, author_candidate, paper_record
):
    paper = deepcopy(paper_record)
    paper.pdf_url = "https://publisher.test/source"
    paper.publication_url = "https://publisher.test/article"

    class NoLLM:
        provider = "none"
        model = ""

        def close(self):
            pass

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/source":
            return httpx.Response(
                200,
                text='<a href="/download/paper">Source PDF</a>',
                headers={"content-type": "text/html"},
            )
        if request.url.path == "/download/paper":
            return httpx.Response(200, content=PDF_BYTES)
        return httpx.Response(200, text="<html>No PDF</html>", headers={"content-type": "text/html"})

    config = replace(app_config, llm=replace(app_config.llm, provider="none", model=""))
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        app = PaperWatch(
            config,
            db=db,
            openalex=FakeOpenAlex([paper]),
            arxiv=None,
            llm=NoLLM(),
            pdf_client=httpx.Client(
                transport=httpx.MockTransport(handler), follow_redirects=True
            ),
        )
        stats = app.run(today=date(2026, 7, 24))
        stored = db.list_papers()[0]
        assert stats.status == "success"
        assert stored["pdf_url"] == "https://publisher.test/download/paper"
        assert stored["pdf_source"] == "publisher_page"
        assert Path(stored["local_pdf_path"]).is_file()


def test_pdf_candidate_collection_includes_openalex_locations(paper_record):
    paper = deepcopy(paper_record)
    paper.pdf_url = "https://repo.test/direct.pdf"
    paper.publication_url = "https://publisher.test/article"
    paper.metadata = {
        "best_oa_location": "malformed",
        "primary_location": {
            "pdf_url": "https://repo.test/direct.pdf",
            "landing_page_url": "https://primary.test/article",
        },
        "locations": [
            None,
            {
                "pdf_url": "https://other.test/file.pdf",
                "landing_page_url": "not-a-url",
            },
        ],
    }
    assert PaperWatch._paper_pdf_candidates(paper) == [
        "https://repo.test/direct.pdf",
        "https://publisher.test/article",
        "https://primary.test/article",
        "https://other.test/file.pdf",
    ]


def test_stub_article_url_and_candidate_edge_cases():
    from paperwatch.pipeline import _stub_link_candidates, _stub_link_is_article

    assert not _stub_link_is_article("not a url")
    assert not _stub_link_is_article("https://arxiv.org/abs/2601.00001")
    assert not _stub_link_is_article("https://example.com/file.pdf")
    assert not _stub_link_is_article("https://example.com/")
    assert not _stub_link_is_article("https://example.com/about/team")
    assert not _stub_link_is_article("https://example.com/research")
    assert _stub_link_is_article("https://example.com/posts/2026/result")
    assert _stub_link_is_article(
        "https://example.com/2026/result",
        blog_url="not a blog url",
    )
    assert not _stub_link_is_article(
        "https://example.com/2026/result",
        blog_url="https://example.com/2026/result",
    )

    entry = {
        "source_url": "https://example.com/posts/2026/result.",
        "normalized_text": (
            "https://example.com/posts/2026/result; "
            "https://example.com/about https://example.com/posts/2026/second! "
            "http://localhost/private"
        ),
        "blog_url": "https://example.com/blog",
    }
    assert _stub_link_candidates(entry) == (
        "https://example.com/posts/2026/result",
        "https://example.com/posts/2026/second",
    )


def test_arxiv_fallback_date_parses_both_formats_and_invalid_values():
    from paperwatch.blogs import ArticleItem
    from paperwatch.pipeline import PaperWatch

    def article(text):
        return ArticleItem(
            external_id="x",
            title="Paper",
            url="https://arxiv.org/abs/1",
            text=text,
            sha256="sha",
        )

    assert PaperWatch._arxiv_fallback_date(None) is None
    assert PaperWatch._arxiv_fallback_date(article("Submitted on 7 Aug 2026")) == "2026-08-07"
    assert PaperWatch._arxiv_fallback_date(article("[v1] Fri, 8 Aug 2025")) == "2025-08-08"
    assert PaperWatch._arxiv_fallback_date(article("Submitted on 99 Aug 2026")) is None
    assert PaperWatch._arxiv_fallback_date(article("No date")) is None


def test_refetch_after_delete_targets_every_recreated_blog_for_automatic_regexp(
    app_config, author_candidate
):
    import hashlib

    from paperwatch.blogs import BlogFetch, FeedItem

    class NoopOpenAlex:
        def close(self):
            pass

    class NoLLM:
        provider = "none"
        model = "none"

        def close(self):
            pass

    items = tuple(
        FeedItem(
            external_id=f"post-{index}",
            title=f"Post {index}",
            url=f"https://example.test/blog/post-{index}",
            published_at=f"2026-08-{index + 1:02d}",
            text=f"Article body marker-{index}",
            sha256=hashlib.sha256(f"Article body marker-{index}".encode()).hexdigest(),
        )
        for index in range(12)
    )
    fetched = BlogFetch(
        method="rss",
        title="Resettable feed",
        status_code=200,
        feed_url="https://example.test/blog/feed.xml",
        feed_items=items,
        source_mode="feed",
    )
    config = replace(
        app_config,
        fetch=replace(app_config.fetch, download_pdfs=False),
        llm=replace(app_config.llm, provider="none", analyze_pdfs=False),
    )

    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        db.add_author_blog(author_candidate.openalex_id, "https://example.test/blog")
        query_id = db.create_saved_query(
            name="Reset regexp",
            prompt="",
            auto_run_blogs=True,
            query_kind="regexp",
            content_scope="blogs",
            regexp_pattern=r"marker-\d+",
        )
        app = PaperWatch(
            config,
            db=db,
            openalex=NoopOpenAlex(),
            arxiv=None,
            llm=NoLLM(),
            pdf_client=httpx.Client(),
        )

        first = RunStats(run_id=db.start_run())
        app._apply_blog_fetch(first, db.list_enabled_blogs()[0], fetched, None)
        assert first.blog_updates_found == 0
        assert len(first.new_blog_entry_ids) == len(items)

        assert db.delete_all_blog_entries() == len(items)

        second = RunStats(run_id=db.start_run())
        app._apply_blog_fetch(second, db.list_enabled_blogs()[0], fetched, None)
        assert second.blog_updates_found == 0
        assert len(second.new_blog_entry_ids) == len(items)
        assert set(second.new_blog_entry_ids) == {
            int(row["id"]) for row in db.list_blog_entries()
        }

        app._run_automatic_queries(second)
        runs = [
            row for row in db.list_query_runs()
            if int(row["query_id"]) == query_id
        ]
        assert len(runs) == len(items)
        assert {int(row["blog_entry_id"]) for row in runs} == set(
            second.new_blog_entry_ids
        )
        assert {row["status"] for row in runs} == {"success"}
        for entry_id in second.new_blog_entry_ids:
            tags = db.classification_results_for_content("blog", entry_id)
            assert len(tags) == 1
            assert tags[0]["tag_name"].startswith("marker-")
