from __future__ import annotations

import json
import threading
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from paperwatch.analysis_jobs import AnalysisBatchSnapshot
from paperwatch.db import Database
from paperwatch.models import AuthorCandidate, PaperRecord, stable_hash
from paperwatch.web import RunController, RunSnapshot, _parse_tristate, _run_pipeline, create_app


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
    assert "Downloaded PDF" in response.text
    assert "Backfill 1 year PDFs" in response.text
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

    response = client.post("/authors/search", data={"name": "Grace Hopper", "limit": "25"})
    assert response.status_code == 200
    assert "Grace Hopper" in response.text
    assert "Search OpenAlex" in response.text
    assert "After tracking" in response.text
    assert 'aria-label="Track Grace Hopper"' in response.text
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
        for index in range(15):
            copy = replace(
                paper_record,
                openalex_id=f"WX{index}",
                doi=f"10.2000/{index}",
                arxiv_id=None,
                title=f"Pagination Paper {index:02d}",
            )
            db.upsert_paper(copy)
    client, _ = make_client(config_path)
    response = client.get("/papers?page=99&page_size=10", follow_redirects=False)
    assert response.status_code == 303
    assert "page=2" in response.headers["location"]
    assert "Previous" in client.get("/papers?page=2&page_size=10").text


def test_paper_detail_and_exports(config_path, seeded_web):
    _, paper_id, second_id, analysis_id, _ = seeded_web
    client, _ = make_client(config_path)
    response = client.get(f"/papers/{paper_id}")
    assert response.status_code == 200
    assert "Fulltext analysis" in response.text
    assert "Key contributions" in response.text
    assert "Raw JSON and usage" in response.text
    assert "Run additional models" in response.text
    assert "Google Scholar" in response.text
    assert "OpenAlex" in response.text
    assert "arXiv" in response.text
    assert "Track author" in response.text
    assert "/?author=A123" in response.text
    assert "https://openalex.org/authors/a123" in response.text
    assert "https://scholar.google.com/scholar?q=author%3A%22Ada+Lovelace%22" in response.text
    assert "https://arxiv.org/search/?query=Ada+Lovelace&amp;searchtype=author" in response.text

    response = client.get(f"/papers/{second_id}")
    assert "No LLM analysis" in response.text
    assert client.get("/papers/999999").status_code == 404

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
