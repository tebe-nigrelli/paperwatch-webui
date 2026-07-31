from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import date
from pathlib import Path

import httpx
import pytest

from paperwatch.db import Database
from paperwatch.models import LLMResponse
from paperwatch.pipeline import PaperWatch
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
        first = app.run(today=date(2026, 7, 24))
        second = app.run(today=date(2026, 7, 24))

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
        stats = app.run(today=date(2026, 7, 24))
        assert stats.status == "success"
        assert stats.papers_analyzed == 2
        assert stats.pdfs_downloaded == 1
        assert db.list_papers()[0]["local_pdf_path"] is not None


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
        stats = app.run(today=date(2026, 7, 24))
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
        stats = app.run(today=date(2026, 7, 24))
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
        stats = app.run(today=date(2026, 7, 24))
        assert stats.status == "success"
        assert stats.papers_analyzed == 0
        assert stats.pdfs_downloaded == 1


def test_complete_openalex_record_skips_arxiv(app_config, author_candidate, paper_record, analysis_result):
    class FailArxiv:
        def by_id(self, value):
            raise AssertionError("arXiv should not be queried")

        def by_title(self, *args):
            raise AssertionError("arXiv should not be queried")

    config = replace(app_config, fetch=replace(app_config.fetch, download_pdfs=False))
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        app = PaperWatch(
            config,
            db=db,
            openalex=FakeOpenAlex([paper_record]),
            arxiv=FailArxiv(),
            llm=FakeLLM(analysis_result),
            pdf_client=httpx.Client(),
        )
        assert app.run(today=date(2026, 7, 24)).status == "success"


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
        stats = app.run(today=date(2026, 7, 24))
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
        stats = app.run(today=date(2026, 7, 24))
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
        stats = app.run(today=date(2026, 7, 24))
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
        stats = app.run(today=date(2026, 7, 24))
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
        stats = app.run(today=date(2026, 7, 24))
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

    config = replace(app_config, fetch=replace(app_config.fetch, download_pdfs=False))
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
            pdf_client=httpx.Client(),
        )
        stats = app.run(today=date(2026, 7, 24))
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
            pdf_client=httpx.Client(),
        )
        stats = app.run(today=date(2026, 7, 24))
        assert stats.status == "partial"
        assert stats.errors_count == 1
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
        stats = app.run(today=date(2026, 7, 24))
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
        normal = app.run(today=date(2026, 7, 31))
        assert normal.papers_analyzed == 0
        assert llm.prompts == []

        # If the PDF first becomes downloadable later, analyze that new download once.
        Path(stored["local_pdf_path"]).unlink()
        late = app.run(today=date(2026, 8, 1))
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

        no_pdf = replace(paper_record, openalex_id="WNOFILE", pdf_url=None)
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
