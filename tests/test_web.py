from __future__ import annotations

import io
import json
import threading
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from paperwatch.analysis_jobs import AnalysisBatchSnapshot
from paperwatch.db import Database
from paperwatch.models import AuthorCandidate, PaperRecord, stable_hash
from paperwatch.web import (
    RunController,
    RunSnapshot,
    _manual_author_names,
    _manual_paper_record,
    _manual_publication_date,
    _manual_source_url,
    _parse_tristate,
    _run_pipeline,
    create_app,
)


class FakeAuthorClient:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.closed = False
        self.search_calls = []
        self.get_calls = []

    def search_authors(self, name: str, limit: int = 10):
        self.search_calls.append((name, limit))
        if self.fail:
            raise RuntimeError("OpenAlex unavailable")
        return [
            AuthorCandidate(
                openalex_id="A777",
                name="Grace Hopper",
                institution="Navy",
                works_count=12,
                cited_by_count=34,
            )
        ]

    def get_author(self, openalex_id: str):
        self.get_calls.append(openalex_id)
        return AuthorCandidate(openalex_id=openalex_id, name="Grace Hopper")

    def close(self):
        self.closed = True


class StubController:
    def __init__(self, starts: list[bool] | None = None, snapshot: RunSnapshot | None = None):
        self.starts = starts or [True]
        self.calls = 0
        self.modes: list[str] = []
        self.state = snapshot or RunSnapshot()

    def start(self, mode: str = "run") -> bool:
        self.modes.append(mode)
        value = self.starts[min(self.calls, len(self.starts) - 1)]
        self.calls += 1
        if value:
            self.state = RunSnapshot(
                running=True,
                mode=mode,
                started_at="2026-07-25T10:00:00+00:00",
                status="running",
            )
        return value

    def snapshot(self) -> RunSnapshot:
        return RunSnapshot(**asdict(self.state))


class StubAnalysisController:
    def __init__(
        self,
        *,
        batch_ids: list[str | None] | None = None,
        snapshot: AnalysisBatchSnapshot | None = None,
        error: Exception | None = None,
    ):
        self.batch_ids = batch_ids or ["batch-1"]
        self.state = snapshot or AnalysisBatchSnapshot()
        self.error = error
        self.calls: list[tuple[int, str, tuple[str, ...]]] = []

    def start(self, paper_id: int, scope: str, profiles: tuple[str, ...]):
        self.calls.append((paper_id, scope, profiles))
        if self.error:
            raise self.error
        value = self.batch_ids[min(len(self.calls) - 1, len(self.batch_ids) - 1)]
        if value:
            self.state = AnalysisBatchSnapshot(
                running=True, batch_id=value, paper_id=paper_id, scope=scope, profiles=profiles
            )
        return value

    def snapshot(self):
        return AnalysisBatchSnapshot(**asdict(self.state))


@pytest.fixture
def seeded_web(config_path, author_candidate, paper_record, analysis_result):
    config = __import__("paperwatch.config", fromlist=["load_config"]).load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        paper_id, _ = db.upsert_paper(paper_record)
        analysis_id = db.save_analysis(
            paper_id=paper_id,
            scope="fulltext",
            provider="ollama",
            model="test-model",
            prompt_version="test-v1",
            content_sha256=stable_hash("fulltext"),
            input_chars=123,
            result=analysis_result,
            usage={"prompt_eval_count": 22},
        )
        task_id = db.create_analysis_tasks(
            batch_id="seeded-analysis",
            paper_id=paper_id,
            scope="fulltext",
            profiles=[("default", "ollama", "test-model")],
        )[0]
        db.finish_analysis_task(task_id, analysis_id=analysis_id)
        second = PaperRecord(
            openalex_id="W999",
            doi=None,
            arxiv_id=None,
            title="A Mathematical Note on Empty Benchmarks",
            abstract=None,
            publication_date="2025-01-02",
            updated_date=None,
            venue="Pure Notes",
            publication_url="https://example.test/note",
            pdf_url=None,
            abstract_source=None,
            pdf_source=None,
            authors=[],
            metadata={},
        )
        second_id, _ = db.upsert_paper(second)
        run_id = db.start_run()
        db.record_error(
            run_id,
            stage="pdf_processing",
            message="broken PDF",
            author_openalex_id="A123",
            paper_identity_key=paper_record.identity_key,
        )
        db.finish_run(
            run_id,
            "partial",
            {
                "authors_checked": 1,
                "papers_found": 2,
                "papers_analyzed": 1,
                "pdfs_downloaded": 0,
                "errors_count": 1,
            },
        )
    pdf = config.paths.pdf_dir / "valid.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF")
    with Database(config.paths.database) as db:
        db.set_local_pdf(paper_id, str(pdf))
    return config, paper_id, second_id, analysis_id, run_id


def valid_pdf_bytes() -> bytes:
    stream = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(stream)
    return stream.getvalue()


def make_client(config_path, *, controller=None, analysis_controller=None, fake=None):
    instances = []

    def factory(_config):
        client = fake or FakeAuthorClient()
        instances.append(client)
        return client

    app = create_app(
        config_path,
        run_controller=controller or StubController(),
        analysis_controller=analysis_controller or StubAnalysisController(),
        author_client_factory=factory,
    )
    return TestClient(app), instances


def test_dashboard_assets_and_security_headers(config_path, seeded_web):
    client, _ = make_client(config_path)
    response = client.get("/")
    assert response.status_code == 200
    assert "Research feed" not in response.text
    assert "Tracked author(s)" in response.text
    assert "OpenAlex topics" in response.text
    assert "PDF file" in response.text
    assert "Minimal Agents" in response.text
    assert "Open PDF" in response.text
    assert "Backfill 1 year PDFs" not in response.text
    assert "Discover" in response.text
    assert "broken PDF" in response.text
    assert response.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in response.headers["content-security-policy"]
    assert client.get("/static/app.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_dashboard_running_and_error_states(config_path, seeded_web):
    running = StubController(snapshot=RunSnapshot(running=True, started_at="now", status="running"))
    client, _ = make_client(config_path, controller=running)
    assert "Fetch in progress" in client.get("/").text

    failed = StubController(snapshot=RunSnapshot(error="model down", status="failed"))
    client, _ = make_client(config_path, controller=failed)
    assert "model down" in client.get("/").text


def test_author_search_add_toggle_remove(config_path, seeded_web):
    client, instances = make_client(config_path)
    author_page = client.get("/authors")
    assert "Ada Lovelace" in author_page.text
    assert "Name<br>University" in author_page.text
    assert "Pause / resume" in author_page.text
    assert "last checked" not in author_page.text
    assert 'placeholder="Add tag"' in author_page.text
    assert ">Add</button>" not in author_page.text
    assert client.get("/authors?q=Analytical+Engine").status_code == 200
    assert "Ada Lovelace" in client.get("/authors?q=Analytical+Engine").text
    assert "Ada Lovelace" not in client.get("/authors?q=unrelated").text

    response = client.post(
        "/authors/search", data={"name": "Grace Hopper", "limit": "25"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/discover?mode=author&q=Grace+Hopper"
    response = client.get(
        "/discover", params={"mode": "author", "q": "Grace Hopper", "limit": "25"}
    )
    assert "Grace Hopper" in response.text
    assert "Find an author on OpenAlex" in response.text
    assert ">Track</button>" in response.text
    assert instances[-1].search_calls == [("Grace Hopper", 25)]
    assert instances[-1].closed

    response = client.post("/authors/A777/add", follow_redirects=False)
    assert response.status_code == 303
    with Database(seeded_web[0].paths.database) as db:
        assert db.get_author("A777")["name"] == "Grace Hopper"

    assert client.post("/authors/A777/disable", follow_redirects=False).status_code == 303
    assert client.post("/authors/A777/enable", follow_redirects=False).status_code == 303
    assert client.post("/authors/A777/remove", follow_redirects=False).status_code == 303
    with Database(seeded_web[0].paths.database) as db:
        archived = db.get_author("A777")
        assert archived is not None and archived["tracked"] == 0
    assert "Grace Hopper" not in client.get("/authors").text
    assert client.post("/authors/A404/enable").status_code == 404
    assert client.post("/authors/A404/disable").status_code == 404
    assert client.post("/authors/A404/remove").status_code == 404


def test_author_search_empty_and_error(config_path, seeded_web):
    client, instances = make_client(config_path)
    response = client.post("/authors/search", data={"name": "", "limit": "999"})
    assert response.status_code == 200
    assert instances == []

    client, instances = make_client(config_path, fake=FakeAuthorClient(fail=True))
    response = client.post("/authors/search", data={"name": "Ada", "limit": "bad"})
    assert "OpenAlex unavailable" in response.text
    assert instances[-1].closed


def test_paper_list_filters_and_invalid_query_values(config_path, seeded_web):
    client, _ = make_client(config_path)
    response = client.get(
        "/papers",
        params={
            "q": "Minimal%_",
            "author": "A123",
            "kind": "machine_learning",
            "scope": "fulltext",
            "has_pdf": "yes",
            "analyzed": "true",
            "date_from": "2026-01-01",
            "date_to": "2026-12-31",
            "sort": "title",
            "page_size": "10",
        },
    )
    assert response.status_code == 200
    assert "0 matching records" in response.text  # escaped wildcard is literal

    response = client.get(
        "/papers",
        params={
            "q": "Minimal",
            "has_pdf": "no",
            "analyzed": "no",
            "kind": "invalid",
            "scope": "bad",
            "sort": "bad",
            "date_from": "not-a-date",
            "date_to": "2026-99-99",
            "page": "-4",
            "page_size": "999",
        },
    )
    assert response.status_code == 200
    assert "No papers match" in response.text

    assert _parse_tristate("0") is False
    assert _parse_tristate("1") is True
    assert _parse_tristate("maybe") is None


def test_paper_pagination_redirect(config_path, seeded_web, paper_record):
    config = seeded_web[0]
    with Database(config.paths.database) as db:
        for index in range(30):
            copy = replace(
                paper_record,
                openalex_id=f"WX{index}",
                doi=f"10.2000/{index}",
                arxiv_id=None,
                title=f"Pagination Paper {index:02d}",
            )
            db.upsert_paper(copy)
    client, _ = make_client(config_path)
    response = client.get("/papers?page=99&page_size=25", follow_redirects=False)
    assert response.status_code == 303
    assert "page=2" in response.headers["location"]
    assert "Previous" in client.get("/papers?page=2&page_size=25").text


def test_paper_detail_and_exports(config_path, seeded_web):
    _, paper_id, second_id, analysis_id, _ = seeded_web
    client, _ = make_client(config_path)
    response = client.get(f"/papers/{paper_id}")
    assert response.status_code == 200
    assert "Summary comparison" not in response.text
    assert "Fulltext analysis" not in response.text
    assert "Key contributions" not in response.text
    assert "Raw JSON and usage" not in response.text
    assert "Recent tasks" in response.text
    assert f'href="/analyses/{analysis_id}"' in response.text
    assert response.text.index("Source metadata") < response.text.index("Recent tasks")
    assert "Run saved prompt" in response.text
    assert "Run additional models" not in response.text
    assert "Google Scholar" in response.text
    assert "OpenAlex" in response.text
    assert "arXiv" in response.text
    assert "External links" in response.text
    assert "Paper actions" in response.text
    assert 'class="paper-reference-layout"' in response.text
    assert 'class="metadata-grid paper-reference-metadata"' in response.text
    assert "Author actions" not in response.text
    assert "Pick an author once" not in response.text
    assert response.text.index("DOI") < response.text.index("External links")
    assert response.text.index("External links") < response.text.index("Paper actions")
    assert "Track author" in response.text
    assert "/?author=A123" in response.text
    assert "https://openalex.org/authors/a123" in response.text
    assert "https://scholar.google.com/scholar?q=author%3A%22Ada+Lovelace%22" in response.text
    assert "https://arxiv.org/search/?query=Ada+Lovelace&amp;searchtype=author" in response.text

    response = client.get(f"/papers/{second_id}")
    assert "Recent tasks" in response.text
    assert "No analysis tasks yet." in response.text
    assert client.get("/papers/999999").status_code == 404

    analysis_page = client.get(f"/analyses/{analysis_id}")
    assert analysis_page.status_code == 200
    assert "Fulltext analysis · test-model" in analysis_page.text
    assert "Key contributions" in analysis_page.text
    assert "Raw JSON and usage" in analysis_page.text
    assert f'href="/papers/{paper_id}"' in analysis_page.text
    assert client.get("/analyses/999999").status_code == 404

    paper_export = client.get(f"/papers/{paper_id}/export.json")
    assert paper_export.status_code == 200
    payload = paper_export.json()
    assert payload["title"].startswith("Minimal Agents")
    assert payload["authors"][0]["name"] == "Ada Lovelace"
    assert payload["analyses"][0]["scope"] == "fulltext"
    assert "attachment" in paper_export.headers["content-disposition"]

    analysis_export = client.get(f"/analyses/{analysis_id}/export.json")
    assert analysis_export.json()["name"].startswith("Minimal Agents")
    assert client.get("/analyses/999999/export.json").status_code == 404
    assert client.get("/papers/999999/export.json").status_code == 404


def test_local_pdf_download_and_path_guard(config_path, seeded_web, tmp_path):
    config, paper_id, second_id, _, _ = seeded_web
    client, _ = make_client(config_path)
    response = client.get(f"/papers/{paper_id}/pdf")
    assert response.status_code == 200
    assert response.content.startswith(b"%PDF")
    assert response.headers["content-disposition"].startswith("inline")
    assert client.get(f"/papers/{second_id}/pdf").status_code == 404
    assert client.get("/papers/99999/pdf").status_code == 404

    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"%PDF")
    with Database(config.paths.database) as db:
        db.set_local_pdf(paper_id, str(outside))
    assert client.get(f"/papers/{paper_id}/pdf").status_code == 404


def test_bulk_exports_preserve_structured_data(config_path, seeded_web):
    client, _ = make_client(config_path)
    page = client.get("/?q=Minimal&page=7&page_size=25")
    assert 'href="/exports/papers.json?q=Minimal"' in page.text
    assert 'href="/exports/papers.csv?q=Minimal"' in page.text
    assert "/exports/papers.json?q=Minimal&amp;page=" not in page.text

    response = client.get("/exports/papers.json?q=Minimal")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["analyses"][0]["main_results"]

    response = client.get("/exports/papers.jsonl")
    lines = [json.loads(line) for line in response.text.splitlines()]
    assert len(lines) == 2
    assert response.headers["content-type"].startswith("application/x-ndjson")

    response = client.get("/exports/papers.csv")
    assert response.status_code == 200
    assert "one_sentence_summary" in response.text
    assert "Minimal Agents" in response.text
    assert "A Mathematical Note" in response.text
    assert client.get("/exports/papers.xml").status_code == 404


def test_run_pages_action_and_status_api(config_path, seeded_web):
    *_, run_id = seeded_web
    controller = StubController(starts=[True, False])
    client, _ = make_client(config_path, controller=controller)
    response = client.get("/runs")
    assert "partial" in response.text
    response = client.get(f"/runs/{run_id}")
    assert "broken PDF" in response.text
    assert client.get("/runs/999999").status_code == 404

    response = client.post("/actions/run", follow_redirects=False)
    assert response.status_code == 303
    assert "Fetch+started" in response.headers["location"]
    assert controller.modes == ["run"]
    response = client.post("/actions/backfill", follow_redirects=False)
    assert "already+running" in response.headers["location"]
    assert controller.modes == ["run", "backfill"]
    status = client.get("/api/run-status").json()
    assert status["running"] is True



def test_paper_author_can_be_tracked_from_summary(config_path, seeded_web):
    config, paper_id, *_ = seeded_web
    client, instances = make_client(config_path)
    response = client.post(f"/papers/{paper_id}/authors/A999/add", follow_redirects=False)
    assert response.status_code == 303
    assert f"/papers/{paper_id}" in response.headers["location"]
    assert instances[-1].get_calls == ["A999"]
    with Database(config.paths.database) as db:
        row = db.get_author("A999")
        assert row is not None and row["tracked"] == 1 and row["enabled"] == 1
    assert client.post("/papers/999999/authors/A999/add").status_code == 404


def test_multi_model_analysis_action_and_status(config_path, seeded_web):
    _, paper_id, *_ = seeded_web
    controller = StubAnalysisController(batch_ids=["batch-ok", None])
    client, _ = make_client(config_path, analysis_controller=controller)
    response = client.post(
        f"/papers/{paper_id}/analyze",
        data={"scope": "fulltext", "model": ["default", "second"]},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "Started+2+model" in response.headers["location"]
    assert controller.calls[-1] == (paper_id, "fulltext", ("default", "second"))
    assert client.get("/api/analysis-status").json()["batch_id"] == "batch-ok"

    response = client.post(
        f"/papers/{paper_id}/analyze",
        data={"scope": "abstract", "model": "default"},
        follow_redirects=False,
    )
    assert "already+running" in response.headers["location"]
    assert client.post(f"/papers/{paper_id}/analyze", data={"scope": "bad"}).status_code == 400

    error_controller = StubAnalysisController(error=ValueError("Select at least one model"))
    error_client, _ = make_client(config_path, analysis_controller=error_controller)
    response = error_client.post(
        f"/papers/{paper_id}/analyze", data={"scope": "abstract"}, follow_redirects=False
    )
    assert "Select+at+least+one+model" in response.headers["location"]


def test_analysis_error_is_rendered(config_path, seeded_web):
    controller = StubAnalysisController(
        snapshot=AnalysisBatchSnapshot(error="GPU unavailable")
    )
    client, _ = make_client(config_path, analysis_controller=controller)
    assert "GPU unavailable" in client.get("/").text

def test_run_controller_success_busy_and_failure(tmp_path):
    gate = threading.Event()

    def runner(_path, mode):
        assert mode == "run"
        gate.wait(1)
        return {"run_id": "12", "status": "partial", "papers_found": 3}

    controller = RunController(tmp_path / "config.toml", runner=runner)
    assert controller.start() is True
    assert controller.start() is False
    assert controller.snapshot().running is True
    gate.set()
    controller.wait()
    snapshot = controller.snapshot()
    assert snapshot.running is False
    assert snapshot.run_id == 12
    assert snapshot.status == "partial"
    assert snapshot.stats["papers_found"] == 3

    broken = RunController(
        tmp_path / "config.toml",
        runner=lambda _path, _mode: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert broken.start()
    broken.wait()
    assert broken.snapshot().error == "boom"
    assert broken.snapshot().status == "failed"


def test_run_controller_records_backfill_mode(tmp_path):
    calls = []

    def runner(_path, mode):
        calls.append(mode)
        return {"run_id": 4, "status": "success"}

    controller = RunController(tmp_path / "config.toml", runner=runner)
    assert controller.start("backfill") is True
    controller.wait()
    snapshot = controller.snapshot()
    assert calls == ["backfill"]
    assert snapshot.mode == "backfill"
    assert snapshot.run_id == 4
    with pytest.raises(ValueError, match="Unknown run mode"):
        controller.start("bad")


def test_run_pipeline_wires_and_closes(monkeypatch, config_path):
    events = []

    class Services:
        db = object()
        openalex = object()
        arxiv = object()
        llm = object()

        def close(self):
            events.append("services-close")

    class Stats:
        run_id = 7
        status = "success"

        def as_dict(self):
            return {"authors_checked": 1, "papers_found": 2}

    class Watch:
        def __init__(self, *args, **kwargs):
            events.append((args, kwargs))

        def run(self, **kwargs):
            events.append(("run", kwargs))
            return Stats()

        def close(self):
            events.append("watch-close")

    monkeypatch.setattr("paperwatch.cli.build_services", lambda config: Services())
    monkeypatch.setattr("paperwatch.pipeline.PaperWatch", Watch)
    result = _run_pipeline(config_path)
    assert result["run_id"] == 7
    assert ("run", {}) in events
    assert events[-2:] == ["watch-close", "services-close"]

    events.clear()
    result = _run_pipeline(config_path, "backfill")
    assert result["run_id"] == 7
    run_event = next(item for item in events if isinstance(item, tuple) and item[0] == "run")
    kwargs = run_event[1]
    assert kwargs["download_only"] is True
    assert kwargs["historical"] is True
    assert kwargs["max_papers_per_author_override"] == 5000
    assert (kwargs["today"] - kwargs["since_override"]).days == 365


def test_run_pipeline_closes_services_when_watch_construction_fails(monkeypatch, config_path):
    events = []

    class Services:
        db = object()
        openalex = object()
        arxiv = object()
        llm = object()

        def close(self):
            events.append("services-close")

    monkeypatch.setattr("paperwatch.cli.build_services", lambda config: Services())
    monkeypatch.setattr(
        "paperwatch.pipeline.PaperWatch",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad prompt")),
    )
    with pytest.raises(RuntimeError, match="bad prompt"):
        _run_pipeline(config_path)
    assert events == ["services-close"]


def test_archived_paper_author_is_reenabled_without_openalex(config_path, seeded_web):
    config, paper_id, *_ = seeded_web
    with Database(config.paths.database) as db:
        assert db.remove_author("A123")
    client, instances = make_client(config_path)
    response = client.post(
        f"/papers/{paper_id}/authors/A123/add", follow_redirects=False
    )
    assert response.status_code == 303
    assert instances == []
    with Database(config.paths.database) as db:
        row = db.get_author("A123")
        assert row["tracked"] == 1 and row["enabled"] == 1


def test_analysis_action_rejects_missing_source_and_paper(config_path, seeded_web):
    _, _, second_id, *_ = seeded_web
    controller = StubAnalysisController()
    client, _ = make_client(config_path, analysis_controller=controller)
    response = client.post(
        f"/papers/{second_id}/analyze",
        data={"scope": "fulltext", "model": "default"},
        follow_redirects=False,
    )
    assert "no+PDF" in response.headers["location"]
    response = client.post(
        f"/papers/{second_id}/analyze",
        data={"scope": "abstract", "model": "default"},
        follow_redirects=False,
    )
    assert "no+abstract" in response.headers["location"]
    assert client.post(
        "/papers/999999/analyze", data={"scope": "abstract", "model": "default"}
    ).status_code == 404
    assert controller.calls == []


def test_author_add_failures_return_actionable_notices(config_path, seeded_web):
    _, paper_id, *_ = seeded_web

    class BrokenGetAuthor(FakeAuthorClient):
        def get_author(self, openalex_id: str):
            raise RuntimeError("OpenAlex unavailable")

    client, _ = make_client(config_path, fake=BrokenGetAuthor())
    response = client.post('/authors/A888/add', follow_redirects=False)
    assert response.status_code == 303
    assert 'Could+not+add+author' in response.headers['location']
    response = client.post(
        f'/papers/{paper_id}/authors/A888/add', follow_redirects=False
    )
    assert response.status_code == 303
    assert 'Could+not+add+author' in response.headers['location']


def test_openalex_topics_render_filter_and_export(config_path, seeded_web):
    client, _ = make_client(config_path)

    response = client.get("/")
    assert response.status_code == 200
    assert '<select name="topic">' in response.text
    assert "Machine Learning (1)" in response.text
    assert 'href="/?topic=T100"' in response.text
    assert "Autonomous Agents" in response.text

    filtered = client.get("/", params={"topic": "T100"})
    assert filtered.status_code == 200
    assert "Minimal Agents for Reliable Machine Learning" in filtered.text
    assert "A Mathematical Note on Empty Benchmarks" not in filtered.text

    detail = client.get(f"/papers/{seeded_web[1]}")
    assert detail.status_code == 200
    assert "OpenAlex topics" in detail.text
    assert "Machine Learning" in detail.text

    exported = client.get(f"/papers/{seeded_web[1]}/export.json")
    assert exported.status_code == 200
    topics = exported.json()["topics"]
    assert [topic["openalex_id"] for topic in topics] == ["T100", "T200"]

    csv_export = client.get("/exports/papers.csv", params={"topic": "T100"})
    assert csv_export.status_code == 200
    assert "topics" in csv_export.text.splitlines()[0]
    assert "Machine Learning; Autonomous Agents" in csv_export.text


def test_invalid_topic_filter_is_ignored(config_path, seeded_web):
    client, _ = make_client(config_path)
    response = client.get("/", params={"topic": "A123"})
    assert response.status_code == 200
    assert "2 matching records" in response.text


def test_backfill_action_starts_dedicated_mode(config_path, seeded_web):
    controller = StubController(starts=[True])
    client, _ = make_client(config_path, controller=controller)
    response = client.post("/actions/backfill", follow_redirects=False)
    assert response.status_code == 303
    assert "One-year+PDF+backfill+started" in response.headers["location"]
    assert controller.modes == ["backfill"]
    page = client.get("/")
    assert "One-year PDF backfill in progress" in page.text


def test_manual_import_metadata_validation_helpers():
    assert _manual_author_names("\n; Jane Doe; John Smith; Ada Lovelace") == [
        "Jane Doe",
        "John Smith",
        "Ada Lovelace",
    ]
    assert _manual_publication_date("") is None
    assert _manual_source_url(" https://example.test/paper ") == "https://example.test/paper"

    for value, message in [
        ("", "At least one author"),
        (";".join(f"Author {index}" for index in range(201)), "At most 200 authors"),
    ]:
        with pytest.raises(ValueError, match=message):
            _manual_author_names(value)
    with pytest.raises(ValueError, match="Source link is required"):
        _manual_source_url("")
    with pytest.raises(ValueError, match="Source link is too long"):
        _manual_source_url("https://example.test/" + "x" * 2000)
    with pytest.raises(ValueError, match="Publication date"):
        _manual_publication_date("not-a-date")

    base = {
        "title": "Manual",
        "authors": "Jane Doe",
        "source_link": "https://example.test/paper",
    }
    for changes, message in [
        ({"title": ""}, "Title is required"),
        ({"title": "x" * 1001}, "Title is too long"),
        ({"abstract": "x" * 1_000_001}, "Abstract is too long"),
        ({"venue": "x" * 501}, "Venue is too long"),
        ({"arxiv_id": "invalid"}, "arXiv ID is invalid"),
    ]:
        with pytest.raises(ValueError, match=message):
            _manual_paper_record(base | changes, filename="paper.pdf")


def test_manual_pdf_import_behaves_like_fetched_paper(config_path, seeded_web):
    client, _ = make_client(config_path)
    form = client.get("/papers/import")
    assert form.status_code == 200
    assert 'enctype="multipart/form-data"' in form.text
    assert 'name="pdf"' in form.text
    assert 'name="title"' in form.text
    assert 'name="authors"' in form.text
    assert 'name="source_link"' in form.text
    assert 'name="abstract"' in form.text

    response = client.post(
        "/papers/import",
        data={
            "title": "A Manually Imported Paper",
            "authors": "Doe, Jane; Smith, John; Doe, Jane",
            "source_link": "https://example.test/manual-paper",
            "abstract": "A manually supplied abstract.",
            "publication_date": "2026-07-30",
            "venue": "Manual Proceedings",
            "doi": "https://doi.org/10.5555/manual",
            "arxiv_id": "2607.99999",
        },
        files={"pdf": ("manual.pdf", valid_pdf_bytes(), "application/pdf")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"].startswith("/papers/")
    assert "Imported+A+Manually+Imported+Paper" in response.headers["location"]
    paper_id = int(response.headers["location"].split("/papers/", 1)[1].split("?", 1)[0])

    config = seeded_web[0]
    with Database(config.paths.database) as db:
        paper = db.get_paper(paper_id)
        assert paper is not None
        assert paper["title"] == "A Manually Imported Paper"
        assert paper["doi"] == "10.5555/manual"
        assert paper["arxiv_id"] == "2607.99999"
        assert paper["publication_url"] == "https://example.test/manual-paper"
        assert paper["pdf_source"] == "manual_upload"
        assert paper["abstract_source"] == "manual"
        assert paper["auto_analyze"] == 1
        assert Path(paper["local_pdf_path"]).is_file()
        authors = db.paper_authors(paper_id)
        assert [row["name"] for row in authors] == ["Doe, Jane", "Smith, John"]
        metadata = json.loads(paper["metadata_json"])
        assert metadata["manual_import"] is True
        assert metadata["original_filename"] == "manual.pdf"
        db.create_saved_query(
            name="Manual review",
            source_type="papers",
            search_query="",
            prompt="Review this paper.",
        )

    detail = client.get(f"/papers/{paper_id}")
    assert detail.status_code == 200
    assert "A Manually Imported Paper" in detail.text
    assert "Open PDF" in detail.text
    assert "Run selected models" not in detail.text
    assert "Run saved prompt" in detail.text
    assert "Manual Proceedings" in detail.text
    assert client.get(f"/papers/{paper_id}/pdf").status_code == 200
    exported = client.get(f"/papers/{paper_id}/export.json").json()
    assert exported["title"] == "A Manually Imported Paper"
    assert exported["authors"][0]["name"] == "Doe, Jane"
    assert "A Manually Imported Paper" in client.get("/").text


def test_manual_pdf_import_validation(config_path, seeded_web):
    client, _ = make_client(config_path)
    missing = client.post(
        "/papers/import",
        data={
            "title": "Missing PDF",
            "authors": "Jane Doe",
            "source_link": "https://example.test/paper",
        },
    )
    assert missing.status_code == 400
    assert "PDF file is required" in missing.text
    assert 'value="Missing PDF"' in missing.text

    bad_source = client.post(
        "/papers/import",
        data={"title": "Bad source", "authors": "Jane Doe", "source_link": "file:///tmp/paper"},
        files={"pdf": ("paper.pdf", valid_pdf_bytes(), "application/pdf")},
    )
    assert bad_source.status_code == 400
    assert "valid http or https URL" in bad_source.text

    bad_pdf = client.post(
        "/papers/import",
        data={
            "title": "Bad PDF",
            "authors": "Jane Doe",
            "source_link": "https://example.test/paper",
        },
        files={"pdf": ("paper.pdf", b"not a pdf", "application/pdf")},
    )
    assert bad_pdf.status_code == 400
    assert "not a complete PDF" in bad_pdf.text

    with Database(seeded_web[0].paths.database) as db:
        rows, _ = db.search_papers(query="Missing PDF", limit=None, offset=0)
        assert rows == []
        rows, _ = db.search_papers(query="Bad PDF", limit=None, offset=0)
        assert rows == []


def test_manual_import_replaces_existing_pdf_safely(config_path, seeded_web):
    client, _ = make_client(config_path)
    payload = {
        "title": "Replacement Test",
        "authors": "Jane Doe",
        "source_link": "https://example.test/replacement",
        "publication_date": "2026-07-30",
        "doi": "10.5555/replacement",
    }
    first = client.post(
        "/papers/import",
        data=payload,
        files={"pdf": ("first.pdf", valid_pdf_bytes(), "application/pdf")},
        follow_redirects=False,
    )
    assert first.status_code == 303
    paper_id = int(first.headers["location"].split("/papers/", 1)[1].split("?", 1)[0])
    config = seeded_web[0]
    with Database(config.paths.database) as db:
        path = Path(db.get_paper(paper_id)["local_pdf_path"])
    path.write_bytes(valid_pdf_bytes() + b"\n% old copy")
    replacement_pdf = valid_pdf_bytes()

    second = client.post(
        "/papers/import",
        data=payload | {"abstract": "Updated metadata."},
        files={"pdf": ("second.pdf", replacement_pdf, "application/pdf")},
        follow_redirects=False,
    )
    assert second.status_code == 303
    second_id = int(second.headers["location"].split("/papers/", 1)[1].split("?", 1)[0])
    assert second_id == paper_id
    assert path.read_bytes() == replacement_pdf
    assert not list(path.parent.glob(path.name + ".backup-*"))
    with Database(config.paths.database) as db:
        assert db.get_paper(paper_id)["abstract"] == "Updated metadata."


def test_manual_import_restores_previous_pdf_when_database_update_fails(
    config_path, seeded_web, monkeypatch
):
    client, _ = make_client(config_path)
    payload = {
        "title": "Restore Test",
        "authors": "Jane Doe",
        "source_link": "https://example.test/restore",
        "publication_date": "2026-07-30",
        "doi": "10.5555/restore",
    }
    first = client.post(
        "/papers/import",
        data=payload,
        files={"pdf": ("first.pdf", valid_pdf_bytes(), "application/pdf")},
        follow_redirects=False,
    )
    assert first.status_code == 303
    paper_id = int(first.headers["location"].split("/papers/", 1)[1].split("?", 1)[0])
    config = seeded_web[0]
    with Database(config.paths.database) as db:
        path = Path(db.get_paper(paper_id)["local_pdf_path"])
    original = path.read_bytes()

    def fail_set_local_pdf(self, imported_paper_id, imported_path):
        raise OSError("database write failed")

    monkeypatch.setattr(Database, "set_local_pdf", fail_set_local_pdf)
    failed = client.post(
        "/papers/import",
        data=payload | {"abstract": "This update should not replace the PDF."},
        files={"pdf": ("second.pdf", valid_pdf_bytes(), "application/pdf")},
    )
    assert failed.status_code == 400
    assert "database write failed" in failed.text
    assert path.read_bytes() == original
    assert not list(path.parent.glob(path.name + ".backup-*"))


def test_recent_search_lists_matches_before_download(
    config_path, seeded_web, paper_record
):
    class PaperSearchClient(FakeAuthorClient):
        def __init__(self):
            super().__init__()
            self.work_searches = []

        def search_works(self, query, limit=10):
            self.work_searches.append((query, limit))
            return [paper_record]

    controller = StubController()
    fake = PaperSearchClient()
    client, _ = make_client(config_path, controller=controller, fake=fake)
    page = client.get("/")
    assert "Search papers by title" not in page.text
    assert "Import PDF" not in page.text
    discover = client.get("/discover")
    assert "Search papers by title" in discover.text
    assert "Find on OpenAlex" in discover.text
    assert 'class="discover-mode-option active" href="/discover?mode=paper"' in discover.text
    assert 'class="discover-mode-option" href="/discover?mode=author"' in discover.text
    assert "Import PDF" in discover.text
    assert "Backfill 1 year" in discover.text

    response = client.get(
        "/discover",
        params={"mode": "paper", "q": "Minimal Agents for Reliable Machine Learning"},
    )
    assert response.status_code == 200
    assert fake.work_searches == [("Minimal Agents for Reliable Machine Learning", 100)]
    assert paper_record.title in response.text
    assert 'method="post" action="/actions/download"' in response.text
    assert f'name="work_id" value="{paper_record.openalex_id}"' in response.text
    assert controller.modes == []


def test_paper_search_error_and_empty_query(config_path, seeded_web):
    class FailingPaperSearchClient(FakeAuthorClient):
        def search_works(self, query, limit=10):
            raise RuntimeError("OpenAlex search unavailable")

    fake = FailingPaperSearchClient()
    client, instances = make_client(config_path, fake=fake)
    empty = client.get("/discover")
    assert empty.status_code == 200
    assert instances == []
    failed = client.get("/discover", params={"mode": "paper", "q": "A paper"})
    assert "OpenAlex search unavailable" in failed.text
    assert fake.closed is True


def test_selected_search_result_starts_download(config_path, seeded_web):
    class DownloadController(StubController):
        def __init__(self, starts=None):
            super().__init__(starts=starts)
            self.queries = []

        def start(self, mode="run", query=None):
            self.modes.append(mode)
            self.queries.append(query)
            value = self.starts[min(self.calls, len(self.starts) - 1)]
            self.calls += 1
            return value

    controller = DownloadController()
    client, _ = make_client(config_path, controller=controller)
    response = client.post(
        "/actions/download",
        data={"work_id": "W123", "title": "A paper", "q": "paper query"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert controller.modes == ["download"]
    assert controller.queries == ["W123"]
    assert response.headers["location"].startswith("/discover?mode=paper&q=paper+query")
    assert "Downloading+A+paper" in response.headers["location"]

    missing = client.post(
        "/actions/download", data={"q": "paper query"}, follow_redirects=False
    )
    assert "Paper+identifier+cannot+be+empty" in missing.headers["location"]

    busy = DownloadController(starts=[False])
    busy_client, _ = make_client(config_path, controller=busy)
    response = busy_client.post(
        "/actions/download", data={"work_id": "W123"}, follow_redirects=False
    )
    assert "already+running" in response.headers["location"]

    plain = StubController()
    plain_client, _ = make_client(config_path, controller=plain)
    response = plain_client.post(
        "/actions/download", data={"work_id": "W123"}, follow_redirects=False
    )
    assert response.status_code == 303
    assert plain.modes == ["download"]

    class RejectDownloadController(DownloadController):
        def start(self, mode="run", query=None):
            raise ValueError("Download rejected")

    rejected_client, _ = make_client(
        config_path, controller=RejectDownloadController()
    )
    response = rejected_client.post(
        "/actions/download", data={"work_id": "W123"}, follow_redirects=False
    )
    assert "Download+rejected" in response.headers["location"]


def test_runs_page_lists_active_summary_and_prompt_links(config_path, seeded_web):
    config, paper_id, *_ = seeded_web
    client, _ = make_client(config_path)
    with Database(config.paths.database) as db:
        db.create_analysis_tasks(
            batch_id="active-summary",
            paper_id=paper_id,
            scope="abstract",
            profiles=[("active", "ollama", "deepseek-r1:8b")],
        )
        query_id = db.create_saved_query(
            name="Active script", source_type="papers", search_query="", prompt="Review"
        )
        db.create_query_runs(
            batch_id="active-query",
            query_id=query_id,
            profiles=[("active", "ollama", "deepseek-r1:8b")],
            context_window_tokens=8192,
            selection={"paper_ids": [paper_id]},
        )
    response = client.get("/runs")
    assert "Currently running" in response.text
    assert "Active script" in response.text
    assert "abstract analysis" in response.text
    assert f'href="/papers/{paper_id}"' in response.text


def test_run_controller_download_and_pipeline_mode(monkeypatch, config_path, tmp_path):
    calls = []

    def runner(path, mode, query):
        calls.append((mode, query))
        return {"run_id": 9, "paper_id": 4, "status": "success"}

    controller = RunController(tmp_path / "config.toml", runner=runner)
    assert controller.start("download", query="W123")
    controller.wait()
    assert calls == [("download", "W123")]
    assert controller.snapshot().paper_id == 4
    with pytest.raises(ValueError, match="cannot be empty"):
        controller.start("download")

    events = []

    class Services:
        db = object()
        openalex = object()
        arxiv = object()
        llm = object()

        def close(self):
            events.append("services-close")

    class Stats:
        run_id = 10
        status = "success"

        def as_dict(self):
            return {"papers_found": 1, "pdfs_downloaded": 1}

    class Watch:
        def __init__(self, *args, **kwargs):
            pass

        def fetch_work(self, work_id):
            events.append(work_id)
            return Stats(), 55

        def close(self):
            events.append("watch-close")

    monkeypatch.setattr("paperwatch.cli.build_services", lambda config: Services())
    monkeypatch.setattr("paperwatch.pipeline.PaperWatch", Watch)
    result = _run_pipeline(config_path, "download", "W123")
    assert result["paper_id"] == 55
    assert events == ["W123", "watch-close", "services-close"]


def test_local_pdf_opens_with_system_application(config_path, seeded_web, monkeypatch):
    _, paper_id, second_id, _, _ = seeded_web
    client, _ = make_client(config_path)
    launched = {}

    monkeypatch.setattr("paperwatch.web.shutil.which", lambda command: "/usr/bin/xdg-open")

    def popen(args, **kwargs):
        launched["args"] = args
        launched["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("paperwatch.web.subprocess.Popen", popen)
    response = client.get(f"/papers/{paper_id}/pdf/open")
    assert response.status_code == 204
    assert launched["args"][0] == "/usr/bin/xdg-open"
    assert launched["args"][1].endswith("valid.pdf")
    assert launched["kwargs"]["start_new_session"] is True
    assert client.get(f"/papers/{second_id}/pdf/open").status_code == 404

    monkeypatch.setattr("paperwatch.web.shutil.which", lambda command: None)
    missing = client.get(f"/papers/{paper_id}/pdf/open")
    assert missing.status_code == 503
    assert "xdg-open is not available" in missing.text

    monkeypatch.setattr("paperwatch.web.shutil.which", lambda command: "/usr/bin/xdg-open")

    def fail_open(*args, **kwargs):
        raise OSError("desktop launcher failed")

    monkeypatch.setattr("paperwatch.web.subprocess.Popen", fail_open)
    failed = client.get(f"/papers/{paper_id}/pdf/open")
    assert failed.status_code == 500
    assert "desktop launcher failed" in failed.text


def test_nav_order_runs_copy_and_builtin_query_protection(config_path, seeded_web):
    config, *_ = seeded_web
    client, _ = make_client(config_path)
    page = client.get("/").text
    assert page.index('href="/dashboard"') > page.index('href="/runs"')
    assert 'id="theme-toggle"' in page

    runs = client.get("/runs").text
    assert "Operations" not in runs
    assert "Currently running" in runs

    with Database(config.paths.database) as db:
        builtin = next(row for row in db.list_saved_queries() if row["is_builtin"])
        builtin_id = int(builtin["id"])
    response = client.post(f"/queries/{builtin_id}/delete", follow_redirects=False)
    assert response.status_code == 303
    assert "cannot+be+deleted" in response.headers["location"]
    with Database(config.paths.database) as db:
        assert db.get_saved_query(builtin_id) is not None


def test_discover_selects_one_search_mode_clamps_limit_and_reports_errors(
    config_path, seeded_web, paper_record, author_candidate
):
    class CombinedClient(FakeAuthorClient):
        def __init__(self):
            super().__init__()
            self.calls = []

        def search_works(self, query, limit=10):
            self.calls.append(("paper", query, limit))
            return [paper_record]

        def search_authors(self, query, limit=10):
            self.calls.append(("author", query, limit))
            return [author_candidate]

    fake = CombinedClient()
    client, _ = make_client(config_path, fake=fake)
    response = client.get(
        "/discover", params={"mode": "paper", "q": "Agents", "limit": "9999"}
    )
    assert response.status_code == 200
    assert "Minimal Agents" in response.text
    assert "Author matches" not in response.text
    assert ">Track</button>" not in response.text
    assert fake.calls == [("paper", "Agents", 100)]

    response = client.get(
        "/discover", params={"mode": "author", "q": "Ada", "limit": "9999"}
    )
    assert response.status_code == 200
    assert "Ada Lovelace" in response.text
    assert "Minimal Agents" not in response.text
    assert "Paper matches" not in response.text
    assert fake.calls[-1] == ("author", "Ada", 100)

    legacy = client.get(
        "/discover", params={"paper_q": "Agents", "author_q": "Ada", "limit": "9999"}
    )
    assert "Minimal Agents" in legacy.text
    assert "Author matches" not in legacy.text
    assert fake.calls[-1] == ("paper", "Agents", 100)

    class BrokenClient(CombinedClient):
        def search_works(self, query, limit=10):
            raise RuntimeError("paper search unavailable")

        def search_authors(self, query, limit=10):
            raise RuntimeError("author search unavailable")

    broken_fake = BrokenClient()
    broken, _ = make_client(config_path, fake=broken_fake)
    response = broken.get(
        "/discover", params={"mode": "paper", "q": "Agents", "limit": "-2"}
    )
    assert "paper search unavailable" in response.text
    assert "Showing up to 1 results" in response.text
    assert broken_fake.calls == []

    response = broken.get(
        "/discover", params={"mode": "author", "q": "Ada", "limit": "-2"}
    )
    assert "author search unavailable" in response.text
    assert "paper search unavailable" not in response.text
