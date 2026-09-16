from __future__ import annotations

import csv
import io
import subprocess
import json
import threading
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from paperwatch.analysis_jobs import AnalysisBatchSnapshot
from paperwatch.blogs import BlogFetch, FeedItem
from paperwatch.db import Database
from paperwatch.models import AuthorCandidate, PaperRecord, stable_hash
from paperwatch.sources.arxiv import ArxivEntry
from paperwatch.web import (
    RunController,
    RunSnapshot,
    _choose_export_directory,
    _desktop_directory_picker,
    _manual_author_names,
    _manual_paper_record,
    _manual_publication_date,
    _manual_source_url,
    _inline_display_text,
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

    def start(self, mode: str = "run", query: str | None = None) -> bool:
        self.modes.append(mode)
        value = self.starts[min(self.calls, len(self.starts) - 1)]
        self.calls += 1
        if value:
            self.state = RunSnapshot(
                running=True,
                mode=mode,
                query=query,
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


def make_client(
    config_path,
    *,
    controller=None,
    analysis_controller=None,
    fake=None,
    arxiv=None,
    blog_factory=None,
):
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
        arxiv_client_factory=(lambda _config: arxiv) if arxiv is not None else None,
        blog_client_factory=blog_factory,
    )
    return TestClient(app), instances


def test_dashboard_assets_and_security_headers(config_path, seeded_web):
    client, _ = make_client(config_path)
    response = client.get("/")
    assert response.status_code == 200
    assert "Research feed" not in response.text
    assert "Tracked author(s)" in response.text
    assert "OpenAlex topics" in response.text
    assert "File present" in response.text
    assert "Minimal Agents" in response.text
    assert "Open PDF" in response.text
    assert "Backfill 1 year PDFs" not in response.text
    assert "Discover" in response.text
    assert "Diagnostics" not in response.text
    assert "broken PDF" not in response.text
    assert response.text.index("content-column-filters") < response.text.index(
        'id="collection-run-label"'
    )
    assert (
        '<details class="advanced-filters collection-runner-details '
        'content-results-runner-details">'
    ) in response.text
    assert response.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in response.headers["content-security-policy"]
    css = client.get("/static/app.css")
    javascript = client.get("/static/app.js")
    assert css.status_code == 200
    assert javascript.status_code == 200
    assert css.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert javascript.headers["cache-control"] == "public, max-age=31536000, immutable"


def test_running_state_uses_header_control_without_feedback_banner(config_path, seeded_web):
    running = StubController(snapshot=RunSnapshot(running=True, started_at="now", status="running"))
    client, _ = make_client(config_path, controller=running)
    page = client.get("/").text
    assert "Stop fetch" in page
    assert "Fetch in progress" not in page

    failed = StubController(snapshot=RunSnapshot(error="model down", status="failed"))
    client, _ = make_client(config_path, controller=failed)
    assert "model down" in client.get("/").text


def test_author_search_add_toggle_remove(config_path, seeded_web):
    client, instances = make_client(config_path)
    author_page = client.get("/authors")
    assert "Ada Lovelace" in author_page.text
    assert "Name<br>University" in author_page.text
    assert 'data-filter-name="enabled"' in author_page.text
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
    assert "No downloaded content matches" in response.text

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


def test_inline_display_text_removes_real_and_escaped_newlines():
    assert _inline_display_text("First\\nSecond\nThird\r\nFourth") == "First Second Third Fourth"


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
    assert "Source metadata" not in response.text
    assert "Copy metadata" not in response.text
    assert 'data-copy-source="paper-source-metadata-copy"' not in response.text
    assert response.text.index("content-detail-toolbar") < response.text.index("Recent tasks")
    assert "Run saved prompt" not in response.text
    assert "Override Defaults" not in response.text
    assert 'data-query-defaults-form' not in response.text
    results_page = client.get("/")
    assert "Run saved prompt" in results_page.text
    assert "Override Defaults" in results_page.text
    assert 'name="paper_processing"' in results_page.text
    assert 'value="whole_text"' in results_page.text
    assert 'value="pdf"' in results_page.text
    assert "Run additional models" not in response.text
    assert "Scholar ↗" in response.text
    assert "OpenAlex ↗" in response.text
    assert "arXiv ↗" in response.text
    assert ">Export</button>" in response.text
    assert f'action="/content/paper/{paper_id}/export"' in response.text
    assert 'class="content-detail-toolbar paper-reference-panel"' in response.text
    assert 'class="content-detail-toolbar-facts" aria-label="Paper details"' in response.text
    assert "<dt>Abstract</dt>" not in response.text
    assert "<dt>PDF</dt>" not in response.text
    assert "Author actions" not in response.text
    assert "Pick an author once" not in response.text
    assert response.text.index("content-detail-toolbar") < response.text.index("author-picker-panel")
    toolbar = response.text.split('class="content-detail-toolbar paper-reference-panel"', 1)[1].split("</nav>", 1)[0]
    assert toolbar.index("Read PDF") < toolbar.index("<dt>DOI</dt>")
    assert toolbar.index("<dt>DOI</dt>") < toolbar.index("<dt>arXiv</dt>")
    assert toolbar.index("<dt>arXiv</dt>") < toolbar.index("<dt>Fetched</dt>")
    source_links = toolbar.index('class="content-detail-source-links content-detail-toolbar-source-links"')
    assert toolbar.index("<dt>Fetched</dt>") < source_links < toolbar.index("Replace PDF")
    fetched = toolbar.split("<dt>Fetched</dt><dd>", 1)[1].split("</dd>", 1)[0]
    assert len(fetched) == 16 and fetched[10] == " " and fetched[13] == ":"
    assert "T" not in fetched
    author_row = response.text.split('class="panel author-picker-panel"', 1)[1].split("</section>", 1)[0]
    assert author_row.index('id="author-status"') < author_row.index('id="author-page"')
    assert author_row.index('id="author-page"') < author_row.index('id="author-scholar"')
    assert 'class="content-detail-author-sources"' not in author_row
    assert "Track" in response.text
    assert "/?content=papers&amp;author=A123" in response.text
    assert "/?content=blogs&amp;author=A123" in response.text
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


def test_selected_content_export_overlay_and_zip_defaults(
    config_path, seeded_web, tmp_path, monkeypatch
):
    config, paper_id, *_ = seeded_web
    client, _ = make_client(config_path)

    page = client.get("/")
    assert 'data-dialog-open="selected-export-dialog"' in page.text
    assert 'name="packaging" value="zip" checked' in page.text
    assert 'name="sources" value="original" checked' in page.text
    assert 'name="include_authors" value="1" checked' in page.text
    assert 'name="include_content_csv" value="1" checked' in page.text
    assert 'data-export-folder-status' in page.text
    assert 'class="icon-button selected-export-button"' in page.text
    assert ">Export…</button>" not in page.text

    with Database(config.paths.database) as db:
        db.set_content_selected("paper", paper_id, True)

    response = client.post(
        "/content/selected/export",
        data={
            "folder": str(tmp_path),
            "packaging": "zip",
            "sources": "original",
            "include_authors": "1",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    archive_path = next(tmp_path.glob("paperwatch-export-*.zip"))
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert any(name.startswith("papers/") and name.endswith(".pdf") for name in names)
        assert {"content.csv", "authors.csv"}.issubset(names)
        content = next(csv.DictReader(io.StringIO(archive.read("content.csv").decode("utf-8-sig"))))
        assert content["title"] == "Minimal Agents for Reliable Machine Learning"
        assert "compact agent architecture" in content["summary"]
        assert content["stored_text"] == "We introduce a compact agent architecture."
        assert json.loads(content["summaries_json"])[0]["one_sentence_summary"]
        assert json.loads(content["details_json"])["doi"] == "10.1000/example"
        authors = archive.read("authors.csv").decode("utf-8-sig")
        assert "Ada Lovelace" in authors
        assert "Alan Turing" in authors

    monkeypatch.setattr("paperwatch.web.extract_pdf_text", lambda _path, max_chars: "Page text")
    simplified_destination = tmp_path / "simplified"
    response = client.post(
        "/content/selected/export",
        data={
            "folder": str(simplified_destination),
            "packaging": "files",
            "sources": "simplified",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    simplified = next(simplified_destination.glob("paperwatch-export-*"))
    assert "Page text" in next((simplified / "papers").glob("*.txt")).read_text()
    assert not (simplified / "authors.csv").exists()

    no_csv_destination = tmp_path / "no-csv"
    response = client.post(
        "/content/selected/export",
        data={
            "folder": str(no_csv_destination),
            "packaging": "files",
            "sources": "original",
            "content_csv_option_present": "1",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    no_csv = next(path for path in no_csv_destination.glob("paperwatch-export-*") if path.is_dir())
    assert not (no_csv / "content.csv").exists()


def test_selected_export_native_folder_picker(config_path, tmp_path, monkeypatch):
    client, _ = make_client(config_path)
    monkeypatch.setattr("paperwatch.web._choose_export_directory", lambda: tmp_path)
    response = client.post("/content/selected/export/folder-picker")
    assert response.status_code == 200
    assert response.json() == {"ok": True, "folder": str(tmp_path.resolve())}

    monkeypatch.setattr("paperwatch.web._choose_export_directory", lambda: None)
    response = client.post("/content/selected/export/folder-picker")
    assert response.status_code == 200
    assert response.json() == {"ok": False, "cancelled": True}

    def failed_picker():
        raise OSError("No picker")

    monkeypatch.setattr("paperwatch.web._choose_export_directory", failed_picker)
    response = client.post("/content/selected/export/folder-picker")
    assert response.status_code == 503
    assert response.json()["message"] == "No picker"


def test_desktop_directory_picker_and_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "paperwatch.web.shutil.which",
        lambda command: f"/usr/bin/{command}" if command in {"kdialog", "zenity"} else None,
    )
    command = _desktop_directory_picker("linux")
    assert command is not None
    assert command[0] == "/usr/bin/kdialog"
    assert command[1] == "--getexistingdirectory"

    monkeypatch.setattr(
        "paperwatch.web.shutil.which",
        lambda command: "/usr/bin/zenity" if command == "zenity" else None,
    )
    assert _desktop_directory_picker("linux")[0] == "/usr/bin/zenity"

    monkeypatch.setattr(
        "paperwatch.web.shutil.which",
        lambda command: "/usr/bin/osascript" if command == "osascript" else None,
    )
    assert _desktop_directory_picker("darwin")[0] == "/usr/bin/osascript"

    monkeypatch.setattr(
        "paperwatch.web.shutil.which",
        lambda command: "C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
        if command == "powershell.exe"
        else None,
    )
    windows = _desktop_directory_picker("win32")
    assert windows is not None
    assert windows[0].endswith("powershell.exe")
    assert windows[1:3] == ["-NoProfile", "-STA"]

    monkeypatch.setattr("paperwatch.web._desktop_directory_picker", lambda: ["picker"])
    monkeypatch.setattr(
        "paperwatch.web.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["picker"], 0, stdout=f"{tmp_path}\n", stderr=""
        ),
    )
    assert _choose_export_directory() == tmp_path.resolve()

    monkeypatch.setattr(
        "paperwatch.web.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["picker"], 1, stdout="", stderr="cancelled"
        ),
    )
    assert _choose_export_directory() is None


def test_selected_content_simplified_blog_files_without_authors(
    config_path, seeded_web, tmp_path
):
    config, paper_id, *_ = seeded_web
    with Database(config.paths.database) as db:
        db.set_content_selected("paper", paper_id, False)
        blog_id = db.add_author_blog("A123", "https://example.test/ada", label="Ada's blog")
        entry_id = db.save_blog_entry(
            blog_id=blog_id,
            title="A useful post",
            content_sha256="post",
            normalized_text="A [linked idea](https://example.test/idea).",
            rendered_html="<p>A linked idea.</p>",
            diff_text="new",
            change_kind="changed",
            previous_entry_id=None,
            source_url="https://example.test/ada/post",
        )
        db.set_content_selected("blog", entry_id, True)
    client, _ = make_client(config_path)

    response = client.post(
        "/content/selected/export",
        data={"folder": str(tmp_path), "packaging": "files", "sources": "simplified"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    output = next(path for path in tmp_path.glob("paperwatch-export-*") if path.is_dir())
    markdown = next((output / "blogs").glob("*.md")).read_text()
    assert "# A useful post" in markdown
    assert "[linked idea](https://example.test/idea)" in markdown
    assert (output / "content.csv").is_file()
    assert not (output / "authors.csv").exists()

    original_destination = tmp_path / "original"
    response = client.post(
        "/content/selected/export",
        data={
            "folder": str(original_destination),
            "packaging": "zip",
            "sources": "original",
            "include_authors": "1",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    with zipfile.ZipFile(next(original_destination.glob("*.zip"))) as archive:
        html_name = next(name for name in archive.namelist() if name.endswith(".html"))
        assert "<p>A linked idea.</p>" in archive.read(html_name).decode()
        assert "Ada Lovelace" in archive.read("authors.csv").decode("utf-8-sig")


def test_run_pages_action_and_status_api(config_path, seeded_web):
    *_, run_id = seeded_web
    controller = StubController(starts=[True, False])
    client, _ = make_client(config_path, controller=controller)
    response = client.get("/runs")
    assert "partial" in response.text
    response = client.get(f"/runs/{run_id}")
    assert "broken PDF" in response.text
    assert client.get("/runs/999999").status_code == 404

    response = client.post(
        "/actions/run",
        headers={"referer": "http://testserver/runs?page=2"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/runs?page=2"
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
    assert 'name="topic" form="content-column-filters"' in response.text
    assert ">Machine Learning<" in response.text
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
    assert "Stop fetch" in page.text
    assert "One-year PDF backfill in progress" not in page.text


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
    assert "Read PDF" in detail.text
    assert "Run selected models" not in detail.text
    assert "Run saved prompt" not in detail.text
    assert "Override Defaults" not in detail.text
    results_page = client.get("/")
    assert "Run saved prompt" in results_page.text
    assert "Override Defaults" in results_page.text
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

    def fail_set_local_pdf(self, imported_paper_id, imported_path, imported_hash=None):
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
    assert 'class="discover-mode-option" href="/discover?mode=blog"' in discover.text
    assert "Import PDF" not in discover.text
    assert "Backfill 1 year" in discover.text
    nav = discover.text.split('<nav aria-label="Primary navigation">', 1)[1].split("</nav>", 1)[0]
    assert nav.index('href="/"') < nav.index('href="/authors"')
    assert nav.index('href="/authors"') < nav.index('href="/queries"')
    assert nav.index('href="/queries"') < nav.index('href="/discover"')
    assert nav.index('href="/discover"') < nav.index('href="/runs"')
    assert nav.index('href="/runs"') < nav.index('href="/data"')
    assert '>Run</a>' in nav

    response = client.get(
        "/discover",
        params={"mode": "paper", "q": "Minimal Agents for Reliable Machine Learning"},
    )
    assert response.status_code == 200
    assert fake.work_searches == [("Minimal Agents for Reliable Machine Learning", 100)]
    assert paper_record.title in response.text
    assert "Open imported source" in response.text
    assert f'/papers/{seeded_web[1]}' in response.text
    assert "discover-paper-downloaded" in response.text
    assert controller.modes == []


@pytest.mark.parametrize(
    "query",
    [
        "2411.00678",
        "arXiv:2411.00678v3",
        "https://arxiv.org/abs/2411.00678",
        "https://arxiv.org/pdf/2411.00678v3.pdf",
        "https://doi.org/10.48550/arXiv.2411.00678",
        "10.48550/arXiv.2411.00678",
    ],
)
def test_discover_resolves_arxiv_formats_via_arxiv_then_openalex(
    config_path, seeded_web, paper_record, query
):
    class PaperSearchClient(FakeAuthorClient):
        def __init__(self):
            super().__init__()
            self.doi_lookups = []
            self.work_searches = []

        def get_work_by_doi(self, doi):
            self.doi_lookups.append(doi)
            raise RuntimeError("not indexed by arXiv DOI")

        def search_works(self, value, limit=10):
            self.work_searches.append((value, limit))
            return [paper_record]

    class FakeArxivClient:
        def __init__(self):
            self.ids = []
            self.closed = False

        def by_id(self, arxiv_id):
            self.ids.append(arxiv_id)
            return ArxivEntry(
                arxiv_id=arxiv_id,
                title=paper_record.title,
                abstract=paper_record.abstract or "Abstract",
                authors=[author.name for author in paper_record.authors],
                published=paper_record.publication_date,
                updated=paper_record.updated_date,
                abstract_url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            )

        def close(self):
            self.closed = True

    openalex = PaperSearchClient()
    arxiv = FakeArxivClient()
    client, _ = make_client(config_path, fake=openalex, arxiv=arxiv)
    response = client.get("/discover", params={"mode": "paper", "q": query})

    assert response.status_code == 200
    assert paper_record.title in response.text
    assert openalex.doi_lookups == ["10.48550/arXiv.2411.00678"]
    assert openalex.work_searches == [(paper_record.title, 100)]
    assert arxiv.ids == ["2411.00678"]
    assert arxiv.closed is True


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


def test_runs_page_exposes_live_fetch_history_counters(config_path, seeded_web):
    config, *_ = seeded_web
    stats = {
        "authors_total": 4,
        "authors_checked": 2,
        "papers_found": 7,
        "papers_downloadable": 5,
        "papers_downloaded": 3,
        "papers_analyzed": 1,
        "pdfs_downloaded": 3,
        "blogs_total": 6,
        "blogs_checked": 4,
        "blogs_with_content": 3,
        "blog_items_available": 9,
        "blog_items_downloaded": 8,
        "blog_updates_found": 2,
        "errors_count": 1,
    }
    with Database(config.paths.database) as db:
        run_id = db.start_run()
        db.update_run_stats(run_id, stats)
    controller = StubController(
        snapshot=RunSnapshot(
            running=True,
            run_id=run_id,
            status="running",
            stats=stats,
            stage="Fetching papers, PDFs, and blogs",
        )
    )
    client, _ = make_client(config_path, controller=controller)
    response = client.get("/runs")
    assert f'data-fetch-history-run-id="{run_id}"' in response.text
    assert 'data-fetch-stat="authors_checked">2</strong>' in response.text
    assert 'data-fetch-stat="blog_items_downloaded">8</strong>' in response.text
    assert client.get("/api/running-status").json()["fetch"]["stats"] == stats
    script = client.get("/static/app.js").text
    assert "updateFetchHistory" in script
    assert "data-fetch-history-run-id" in script


def test_run_controller_accepts_live_fetch_stats(tmp_path):
    published = threading.Event()
    release = threading.Event()

    def runner(path, mode, query, progress, cancel_event):
        progress(
            "run_stats",
            json.dumps(
                {
                    "run_id": 17,
                    "authors_total": 5,
                    "authors_checked": 2,
                    "papers_found": 9,
                }
            ),
        )
        published.set()
        release.wait(2)
        return {"run_id": 17, "status": "success", "papers_found": 9}

    controller = RunController(tmp_path / "config.toml", runner=runner)
    assert controller.start()
    assert published.wait(1)
    snapshot = controller.snapshot()
    assert snapshot.run_id == 17
    assert snapshot.stats == {
        "authors_total": 5,
        "authors_checked": 2,
        "papers_found": 9,
    }
    release.set()
    controller.wait()


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
    assert "Desktop opener" in missing.text
    assert "xdg-open" in missing.text

    monkeypatch.setattr("paperwatch.web.shutil.which", lambda command: "/usr/bin/xdg-open")

    def fail_open(*args, **kwargs):
        raise OSError("desktop launcher failed")

    monkeypatch.setattr("paperwatch.web.subprocess.Popen", fail_open)
    failed = client.get(f"/papers/{paper_id}/pdf/open")
    assert failed.status_code == 500
    assert "desktop launcher failed" in failed.text


def test_windows_desktop_open_uses_startfile(tmp_path, monkeypatch):
    from paperwatch.web import _open_desktop_path

    opened = []
    target = tmp_path / "paper.pdf"
    target.write_bytes(b"%PDF-1.4\n%%EOF\n")
    monkeypatch.setattr("paperwatch.web.sys.platform", "win32")
    monkeypatch.setattr("paperwatch.web.os.startfile", lambda path: opened.append(path), raising=False)

    _open_desktop_path(target)

    assert opened == [str(target.resolve())]


def test_macos_desktop_opener(monkeypatch):
    looked_up = []

    def which(command):
        looked_up.append(command)
        return "/usr/bin/open"

    monkeypatch.setattr("paperwatch.web.shutil.which", which)
    from paperwatch.web import _desktop_opener

    assert _desktop_opener("darwin") == "/usr/bin/open"
    assert looked_up == ["open"]


def test_nav_order_runs_copy_and_builtin_query_protection(config_path, seeded_web):
    config, *_ = seeded_web
    client, _ = make_client(config_path)
    page = client.get("/").text
    assert 'href="/runs"' in page
    assert 'href="/dashboard"' not in page
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


def test_paper_detail_shows_structured_prompt_summary(config_path, seeded_web, analysis_result):
    _, paper_id, *_ = seeded_web
    client, _ = make_client(config_path)
    page = client.get(f"/papers/{paper_id}")
    assert page.status_code == 200
    assert "Prompt summary" in page.text
    assert analysis_result.one_sentence_summary in page.text
    assert "Open full analysis" in page.text


def test_query_list_hides_prompt_and_edit_page_reveals_it(config_path):
    config = __import__("paperwatch.config", fromlist=["load_config"]).load_config(config_path)
    with Database(config.paths.database) as db:
        query = db.list_saved_queries()[0]
        query_id = int(query["id"])
        prompt_prefix = "Summarize each selected paper"
    client, _ = make_client(config_path)
    listing = client.get("/queries")
    assert listing.status_code == 200
    assert prompt_prefix not in listing.text
    assert "<th>Name</th>" in listing.text
    edit = client.get(f"/queries/{query_id}/edit")
    assert edit.status_code == 200
    assert prompt_prefix in edit.text
    assert client.get("/queries/999999/edit").status_code == 404


def test_run_controller_reports_progress_and_accepts_cancellation(tmp_path):
    entered = threading.Event()

    def runner(_path, mode, query, progress, cancel_event):
        assert mode == "run" and query is None
        progress("stage", "Fetching papers, PDFs, and blogs")
        progress("progress", "2/7")
        progress("message", "Checked two authors")
        entered.set()
        assert cancel_event.wait(1)
        return {"run_id": 44, "status": "cancelled", "papers_found": 2}

    controller = RunController(tmp_path / "config.toml", runner=runner)
    assert controller.start("run")
    assert entered.wait(1)
    snapshot = controller.snapshot()
    assert snapshot.stage == "Fetching papers, PDFs, and blogs"
    assert (snapshot.progress_current, snapshot.progress_total) == (2, 7)
    assert snapshot.message == "Checked two authors"
    controller._progress("progress", "invalid")
    controller._progress("progress", "bad/value")
    assert (controller.snapshot().progress_current, controller.snapshot().progress_total) == (0, 0)
    assert controller.cancel() is True
    controller.wait()
    snapshot = controller.snapshot()
    assert snapshot.cancelled is True
    assert snapshot.status == "cancelled"
    assert snapshot.stage == "Cancelled"
    assert controller.cancel() is False


def test_fetch_button_toggles_to_cancel_and_running_page_is_live(config_path):
    class ToggleController:
        def __init__(self):
            self.cancelled = False
            self.state = RunSnapshot(
                running=True,
                mode="run",
                started_at="2026-08-01T07:00:00+00:00",
                status="running",
                progress_current=3,
                progress_total=9,
                stage="Fetching papers, PDFs, and blogs",
                message="Checked author Tri Dao",
            )

        def snapshot(self):
            return RunSnapshot(**asdict(self.state))

        def cancel(self):
            self.cancelled = True
            self.state.cancel_requested = True
            return True

        def start(self, mode="run"):
            return False

    controller = ToggleController()
    app = create_app(
        config_path,
        run_controller=controller,
        analysis_controller=StubAnalysisController(),
    )
    client = TestClient(app)
    dashboard = client.get("/")
    assert "Stop fetch" in dashboard.text
    assert "Request cancellation after the current fetch or model step." in dashboard.text
    runs = client.get("/runs")
    assert "3/9" in runs.text
    assert "Checked author Tri Dao" in runs.text
    status = client.get("/api/running-status").json()
    assert status["fetch"]["progress_current"] == 3
    assert status["fetch"]["progress_total"] == 9
    response = client.post("/actions/run", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/"
    assert controller.cancelled is True
    assert "Fetch cancellation requested" not in client.get("/").text


def test_discover_paper_preview_and_async_import_state(
    config_path, seeded_web, paper_record
):
    novel = replace(
        paper_record,
        openalex_id="W888",
        doi="10.1000/novel",
        arxiv_id=None,
        title="A Novel Discover Preview Paper",
    )

    class PaperDiscoveryClient(FakeAuthorClient):
        def search_works(self, query, limit=10):
            assert query == "Novel preview"
            return [novel]

        def get_work(self, work_id):
            assert work_id == "W888"
            return novel

    controller = StubController()
    client, _ = make_client(
        config_path, controller=controller, fake=PaperDiscoveryClient()
    )
    results = client.get("/discover", params={"mode": "paper", "q": "Novel preview"})
    assert results.status_code == 200
    assert "/discover/paper?work_id=W888" in results.text
    assert 'action="/api/discover/paper-import"' in results.text
    assert ">Import</button>" in results.text

    preview = client.get(
        "/discover/paper", params={"work_id": "W888", "q": "Novel preview"}
    )
    assert preview.status_code == 200
    assert "OpenAlex preview" in preview.text
    assert novel.abstract in preview.text
    assert 'action="/api/discover/paper-import"' in preview.text

    started = client.post("/api/discover/paper-import", data={"work_id": "W888"})
    assert started.status_code == 202
    assert started.json() == {"state": "importing"}
    assert controller.state.query == "W888"
    assert client.get(
        "/api/discover/paper-import-status", params={"work_id": "W888"}
    ).json() == {"state": "importing"}

    config = seeded_web[0]
    pdf = config.paths.pdf_dir / "discover-novel.pdf"
    pdf.write_bytes(valid_pdf_bytes())
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(novel)
        db.set_local_pdf(paper_id, str(pdf))
    controller.state = RunSnapshot(
        running=False,
        mode="download",
        query="W888",
        paper_id=paper_id,
        status="success",
    )
    finished = client.get(
        "/api/discover/paper-import-status", params={"work_id": "W888"}
    )
    assert finished.json() == {
        "state": "imported",
        "paper_id": paper_id,
        "url": f"/papers/{paper_id}",
    }

    refreshed = client.get(
        "/discover", params={"mode": "paper", "q": "Novel preview"}
    )
    assert "discover-paper-downloaded" in refreshed.text
    assert "Open imported source" in refreshed.text


def test_discover_blog_previews_and_imports_source_or_single_page(
    config_path, seeded_web, monkeypatch
):
    source_url = "https://example.com/blog"
    post_url = "https://example.com/blog/post-one"
    feed_item = FeedItem(
        external_id="post-one",
        title="Post One",
        url=post_url,
        published_at="2026-08-01",
        text="A useful preview of the first post.",
        sha256=stable_hash("A useful preview of the first post."),
    )

    def fake_fetch_blog(_client, row):
        if row["source_mode"] == "article":
            text = f"Title: Post One\nURL: {post_url}\n\nComplete post body."
            return BlogFetch(
                method="snapshot",
                title="Post One",
                status_code=200,
                snapshot_text=text,
                snapshot_sha256=stable_hash(text),
                snapshot_published_at="2026-08-01",
                source_mode="article",
            )
        return BlogFetch(
            method="rss",
            title="Example Blog",
            status_code=200,
            feed_url="https://example.com/feed.xml",
            discovery_checked=True,
            feed_items=(feed_item,),
            source_mode="feed",
        )

    class DummyBlogClient:
        def close(self):
            pass

    monkeypatch.setattr("paperwatch.web.fetch_blog", fake_fetch_blog)
    client, _ = make_client(
        config_path, blog_factory=lambda _config: DummyBlogClient()
    )

    preview = client.get("/discover", params={"mode": "blog", "q": source_url})
    assert preview.status_code == 200
    assert 'href="/discover?mode=blog"' in preview.text
    assert "Example Blog" in preview.text
    assert "Post One" in preview.text
    assert "Import whole blog" in preview.text
    assert "Review &amp; import" in preview.text

    whole_form = client.get("/discover/blog/import", params={"source": source_url})
    assert whole_form.status_code == 200
    assert "Import whole blog" in whole_form.text
    assert "Ada Lovelace" in whole_form.text

    whole_import = client.post(
        "/discover/blog/import",
        data={
            "source": source_url,
            "scope": "whole",
            "author_openalex_id": "A123",
            "label": "Example research blog",
            "tags": "research, example",
        },
        follow_redirects=False,
    )
    assert whole_import.status_code == 303
    assert whole_import.headers["location"].startswith("/blogs?")
    with Database(seeded_web[0].paths.database) as db:
        blogs = db.author_blogs("A123")
        assert any(row["url"] == source_url for row in blogs)
        assert db.count_blog_entries() >= 1

    entry_form = client.get(
        "/discover/blog/import", params={"source": source_url, "entry": post_url}
    )
    assert entry_form.status_code == 200
    assert "Import page" in entry_form.text
    assert "Complete post body" in entry_form.text

    entry_import = client.post(
        "/discover/blog/import",
        data={
            "source": source_url,
            "entry": post_url,
            "scope": "entry",
            "new_author_name": "Example Team",
            "notes": "Imported from Discover",
        },
        follow_redirects=False,
    )
    assert entry_import.status_code == 303
    with Database(seeded_web[0].paths.database) as db:
        created = db.find_authors_by_name("Example Team")
        assert created
        blogs = db.author_blogs(str(created[0]["openalex_id"]))
        assert any(row["url"] == post_url for row in blogs)


def test_import_paper_accepts_arxiv_link_and_downloads_pdf(
    config_path, seeded_web, monkeypatch
):
    class FakeArxivClient:
        def __init__(self):
            self.ids: list[str] = []
            self.closed = False

        def by_id(self, arxiv_id: str):
            self.ids.append(arxiv_id)
            return ArxivEntry(
                arxiv_id=arxiv_id,
                title="Direct arXiv Import",
                abstract="Imported from an arXiv URL.",
                authors=["Ada Lovelace", "Grace Hopper"],
                published="2026-07-01",
                updated="2026-07-02",
                abstract_url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            )

        def close(self):
            self.closed = True

    arxiv = FakeArxivClient()
    client, _ = make_client(config_path, arxiv=arxiv)
    source = "https://arxiv.org/abs/2607.12345v2"
    selected = client.get("/papers/import", params={"q": source})
    assert selected.status_code == 200
    assert "Direct arXiv Import" in selected.text
    assert f'name="remote_reference" value="{source}"' in selected.text

    def fake_download(url, destination, **_kwargs):
        assert url == "https://arxiv.org/pdf/2607.12345"
        destination.write_bytes(valid_pdf_bytes())
        return True

    monkeypatch.setattr("paperwatch.web.download_pdf", fake_download)
    imported = client.post(
        "/papers/import",
        data={"remote_reference": source, "tags": "linked"},
        follow_redirects=False,
    )
    assert imported.status_code == 303
    paper_id = int(imported.headers["location"].split("/papers/", 1)[1].split("?", 1)[0])
    with Database(seeded_web[0].paths.database) as db:
        paper = db.get_paper(paper_id)
        assert paper["arxiv_id"] == "2607.12345"
        assert Path(paper["local_pdf_path"]).is_file()
        assert "linked" in {row["name"] for row in db.paper_tags(paper_id)}
    assert arxiv.ids == ["2607.12345", "2607.12345"]
    assert arxiv.closed is True


def test_import_paper_search_results_are_clickable(config_path, seeded_web, paper_record):
    class PaperImportClient(FakeAuthorClient):
        def search_works(self, query, limit=10):
            assert query == "Minimal Agents"
            assert limit == 25
            return [paper_record]

        def get_work(self, work_id):
            assert work_id == paper_record.openalex_id
            return paper_record

    client, _ = make_client(config_path, fake=PaperImportClient())
    results = client.get("/papers/import", params={"q": "Minimal Agents"})
    assert results.status_code == 200
    target = f"/papers/import?source={paper_record.openalex_id}"
    assert target in results.text
    selected = client.get(target)
    assert selected.status_code == 200
    assert paper_record.title in selected.text
    assert f'name="remote_reference" value="{paper_record.openalex_id}"' in selected.text


def test_discover_paper_import_api_covers_validation_existing_busy_and_status_states(
    config_path, seeded_web
):
    config, paper_id, *_ = seeded_web
    client, _ = make_client(config_path)
    assert client.post("/api/discover/paper-import", data={"work_id": "bad"}).status_code == 400
    assert client.get("/api/discover/paper-import-status?work_id=bad").status_code == 400

    imported = client.post("/api/discover/paper-import", data={"work_id": "W456"})
    assert imported.json() == {"state": "imported", "paper_id": paper_id, "url": f"/papers/{paper_id}"}
    assert client.get("/api/discover/paper-import-status?work_id=W456").json()["state"] == "imported"

    same = StubController(snapshot=RunSnapshot(running=True, mode="download", query="W777"))
    same_client, _ = make_client(config_path, controller=same)
    assert same_client.post("/api/discover/paper-import", data={"work_id": "W777"}).status_code == 202

    busy = StubController(snapshot=RunSnapshot(running=True, mode="run", query=None))
    busy_client, _ = make_client(config_path, controller=busy)
    assert busy_client.post("/api/discover/paper-import", data={"work_id": "W777"}).status_code == 409

    refused = StubController(starts=[False])
    refused_client, _ = make_client(config_path, controller=refused)
    assert refused_client.post("/api/discover/paper-import", data={"work_id": "W777"}).status_code == 409

    class InvalidStart(StubController):
        def start(self, mode="run", query=None):
            raise ValueError("invalid download")

    invalid_client, _ = make_client(config_path, controller=InvalidStart())
    response = invalid_client.post("/api/discover/paper-import", data={"work_id": "W777"})
    assert response.status_code == 400
    assert response.json()["error"] == "invalid download"

    failed = StubController(
        snapshot=RunSnapshot(
            running=False,
            mode="download",
            query="W777",
            status="failed",
            error="network failed",
        )
    )
    failed_client, _ = make_client(config_path, controller=failed)
    assert failed_client.get("/api/discover/paper-import-status?work_id=W777").json() == {
        "state": "failed",
        "error": "network failed",
    }
    idle_client, _ = make_client(config_path, controller=StubController())
    assert idle_client.get("/api/discover/paper-import-status?work_id=W777").json() == {"state": "idle"}


def test_discover_paper_preview_reports_invalid_ids_and_client_failures(config_path):
    client, _ = make_client(config_path)
    assert client.get("/discover/paper?work_id=nope").status_code == 400

    class FailingWorkClient(FakeAuthorClient):
        def get_work(self, _work_id):
            raise RuntimeError("work vanished")

    failing, instances = make_client(config_path, fake=FailingWorkClient())
    response = failing.get("/discover/paper?work_id=W777")
    assert response.status_code == 404
    assert "work vanished" in response.text
    assert instances[0].closed


def test_blog_preview_and_discovered_import_cover_duplicates_and_metadata(
    config_path, seeded_web, monkeypatch
):
    from paperwatch.blogs import ArticleItem, ArticleLink
    from paperwatch.web import _blog_preview_entries

    article = ArticleItem(
        external_id="article-1",
        title="Full article",
        url="https://example.com/blog/post-1",
        text="Full article body",
        sha256="article-sha",
        rendered_html="<p>Full article body</p>",
        published_at="2026-08-01",
    )
    second_article = ArticleItem(
        external_id="article-2",
        title="Second article",
        url="https://example.com/blog/post-2",
        text="Second body",
        sha256="second-sha",
    )
    fetched = BlogFetch(
        method="rss",
        title="Example blog",
        status_code=200,
        feed_url="https://example.com/blog/feed.xml",
        discovery_checked=True,
        feed_items=(
            FeedItem(
                external_id="feed-1",
                title="Feed title",
                url=article.url,
                published_at="2026-07-31",
                text="Feed excerpt",
                sha256="feed-sha",
            ),
            FeedItem(
                external_id="feed-duplicate-url",
                title="Duplicate URL",
                url=article.url,
                published_at=None,
                text="Ignored duplicate preview",
                sha256="duplicate-sha",
            ),
        ),
        article_items=(article, second_article),
        discovered_articles=(
            ArticleLink("Already fetched", second_article.url, "duplicate"),
            ArticleLink("Third article", "https://example.com/blog/post-3", "Third summary"),
        ),
        source_mode="feed",
        diagnostics=("feed chosen",),
        article_cursor_next=4,
    )
    previews = _blog_preview_entries(fetched, "https://example.com/blog")
    assert [item["url"] for item in previews] == [
        article.url,
        second_article.url,
        "https://example.com/blog/post-3",
    ]
    assert previews[0]["title"] == "Full article"
    assert previews[0]["summary"] == "Full article body"

    monkeypatch.setattr("paperwatch.web._fetch_blog_preview", lambda *_args, **_kwargs: fetched)

    class Closable:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    clients = []

    def blog_factory(_config):
        item = Closable()
        clients.append(item)
        return item

    client, _ = make_client(config_path, blog_factory=blog_factory)
    form = {
        "source": "https://example.com/blog",
        "scope": "whole",
        "author_openalex_id": "A123",
        "label": "Imported blog",
        "relationship": "author",
        "tags": "research, ai",
        "notes": "Imported from Discover",
    }
    first = client.post("/discover/blog/import", data=form, follow_redirects=False)
    assert first.status_code == 303
    assert "stored+4+entries" in first.headers["location"]
    second = client.post("/discover/blog/import", data=form, follow_redirects=False)
    assert second.status_code == 303
    assert "stored+0+entries" in second.headers["location"]
    assert clients and all(item.closed for item in clients)

    config = __import__("paperwatch.config", fromlist=["load_config"]).load_config(config_path)
    with Database(config.paths.database) as db:
        sources = db.author_blogs("A123")
        imported = next(row for row in sources if row["url"] == "https://example.com/blog")
        assert imported["label"] == "Imported blog"
        assert {row["name"] for row in db.blog_source_tags(int(imported["id"]))} == {"research", "ai"}
        assert imported["notes"] == "Imported from Discover"


def test_blog_import_form_and_post_validation_paths(config_path, seeded_web, monkeypatch):
    client, _ = make_client(config_path)
    assert client.get("/discover/blog/import").status_code == 400
    assert client.get("/discover/blog/import?source=http://localhost/blog").status_code == 400

    monkeypatch.setattr("paperwatch.web._fetch_blog_preview", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("preview failed")))
    response = client.get("/discover/blog/import?source=https://example.com/blog")
    assert response.status_code == 200
    assert "preview failed" in response.text

    cases = [
        {"source": "https://example.com/blog", "scope": "bad", "author_openalex_id": "A123"},
        {"source": "https://example.com/blog", "scope": "entry", "author_openalex_id": "A123"},
        {"source": "https://example.com/blog", "scope": "whole"},
        {"source": "https://example.com/blog", "scope": "whole", "author_openalex_id": "A404"},
    ]
    for form in cases:
        result = client.post("/discover/blog/import", data=form, follow_redirects=False)
        assert result.status_code == 303
        assert "level=error" in result.headers["location"]


def test_web_helper_edge_cases_for_arxiv_preview_query_and_progress(config_path, paper_record):
    from starlette.datastructures import FormData
    from paperwatch.config import load_config
    from paperwatch.web import PaperFilters, _blog_preview_entries, _query_execution_form, _search_arxiv_identifier

    filters = PaperFilters(query="agents", author_ids=("A1", "A2"), page=3, page_size=50, read=False)
    assert "q=agents" in filters.query_string
    assert "author=A1" in filters.query_string and "page=3" in filters.query_string

    controller = RunController(config_path, runner=lambda *_args: {})
    controller._progress("run_stats", "not json")
    controller._progress("run_stats", "[]")
    controller._progress("run_stats", '{"run_id":"7","papers_found":3,"ignored":"x"}')
    snapshot = controller.snapshot()
    assert snapshot.run_id == 7
    assert snapshot.stats == {"papers_found": 3}

    class OA:
        def __init__(self):
            self.searches = []

        def search_works(self, query, limit=10):
            self.searches.append((query, limit))
            return [paper_record]

        def get_work_by_doi(self, _doi):
            raise RuntimeError("not linked")

    class AX:
        def __init__(self, entry):
            self.entry = entry

        def by_id(self, _identifier):
            return self.entry

    oa = OA()
    assert _search_arxiv_identifier("plain title", openalex=oa, arxiv=AX(None), limit=4) == [paper_record]
    assert _search_arxiv_identifier("2411.00678", openalex=oa, arxiv=AX(None), limit=4) == []
    entry = ArxivEntry(
        arxiv_id="2411.00678",
        title="Resolved title",
        abstract="a",
        authors=("Ada",),
        published="2024-11-01",
        updated=None,
        doi=None,
        abstract_url="https://arxiv.org/abs/2411.00678",
        pdf_url="https://arxiv.org/pdf/2411.00678.pdf",
    )

    class OAFails(OA):
        def search_works(self, query, limit=10):
            raise RuntimeError("search failed")

    assert _search_arxiv_identifier("2411.00678", openalex=OAFails(), arxiv=AX(entry), limit=4) == []

    empty_preview = _blog_preview_entries(
        BlogFetch(method="html", title="Snapshot", status_code=200, snapshot_text=" Snapshot body "),
        "https://example.com/blog",
    )
    assert empty_preview[0]["title"] == "Snapshot"

    config = load_config(config_path)
    # Saved queries no longer persist a model. Legacy duplicate model fields are
    # ignored and execution resolves the current config default at run time.
    form = _query_execution_form(
        FormData([("model", "default"), ("model", "second")]),
        replace(config, llm_profiles=(replace(config.llm, profile="second"),)),
    )
    assert form["model_profiles"] == ()
    with pytest.raises(ValueError, match="Codex"):
        _query_execution_form(FormData([("paper_processing", "pdf")]), config)


def test_author_csv_import_exercises_optional_identity_state_tags_and_notes(
    config_path, seeded_web
):
    client, _ = make_client(config_path)
    payload = (
        "Author Name,Institution,OpenAlex ID,Starred,Enabled,Tags,Notes\n"
        "CSV OpenAlex,Lab,A987654321,yes,no,vision; ml,Known identity\n"
        "CSV Manual,Independent,,no,yes,systems,Local identity\n"
        ",Ignored,,,,,\n"
    ).encode()
    response = client.post(
        "/authors/import",
        files={"csv_file": ("authors.csv", payload, "text/csv")},
        data={"import_tags": "imported"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "Imported+2+author+rows" in response.headers["location"]

    config = __import__("paperwatch.config", fromlist=["load_config"]).load_config(config_path)
    with Database(config.paths.database) as db:
        known = db.get_author("A987654321")
        assert known is not None and known["starred"] == 1 and known["enabled"] == 0
        assert known["notes"] == "Known identity"
        assert set(db.list_authors_with_tags(query="A987654321")[0]["tags"].split(", ")) == {"vision", "ml", "imported"}
        manual = db.find_authors_by_name("CSV Manual")[0]
        assert str(manual["openalex_id"]).startswith("LOCAL-")
        assert manual["enabled"] == 1
        assert manual["notes"] == "Local identity"


def test_query_classification_delete_and_content_action_response_modes(
    config_path, seeded_web
):
    config, paper_id, *_ = seeded_web
    with Database(config.paths.database) as db:
        query_id = db.create_saved_query(
            name="Disposable labels",
            prompt="Classify",
            query_kind="classification",
            content_scope="papers",
            label_class="Fit",
            label_values=[
                {"value": "Yes", "pattern": "yes"},
                {"value": "No", "pattern": "no"},
            ],
        )
    client, _ = make_client(config_path)
    missing = client.post("/queries/999999/classifications/delete", follow_redirects=False)
    assert missing.status_code == 404
    deleted = client.post(f"/queries/{query_id}/classifications/delete", follow_redirects=False)
    assert deleted.status_code == 303
    assert "Removed+0+classification+results" in deleted.headers["location"]

    async_read = client.post(
        f"/content/paper/{paper_id}/read",
        data={"read": "1"},
        headers={"X-Paperwatch-Async": "1"},
    )
    assert async_read.json() == {"ok": True, "read": True}
    select_results = client.post(
        "/content/select-results",
        data={"content": "papers", "selected": "1"},
        follow_redirects=False,
    )
    assert select_results.status_code == 303
    selected_update = client.post("/content/selected/read", follow_redirects=False)
    assert selected_update.status_code == 303


def test_upload_staging_and_saved_query_form_defensive_edges(config_path, tmp_path):
    import asyncio
    from starlette.datastructures import FormData, UploadFile
    from paperwatch.config import load_config
    from paperwatch.web import _query_execution_form, _saved_query_run_options, _stage_uploaded_pdf

    with pytest.raises(Exception, match="PDF file is empty"):
        asyncio.run(
            _stage_uploaded_pdf(
                UploadFile(file=io.BytesIO(b""), filename="empty.pdf"),
                directory=tmp_path,
                max_bytes=10,
            )
        )
    with pytest.raises(Exception, match="exceeds maximum size"):
        asyncio.run(
            _stage_uploaded_pdf(
                UploadFile(file=io.BytesIO(b"123456"), filename="large.pdf"),
                directory=tmp_path,
                max_bytes=5,
            )
        )

    config = load_config(config_path)

    class Existing:
        regexp_debug_text = "saved regexp debug"
        label_debug_text = "saved label debug"

    values = _query_execution_form(FormData(), config, existing_query=Existing())
    assert values["regexp_debug_text"] == "saved regexp debug"
    assert values["label_debug_text"] == "saved label debug"

    with Database(config.paths.database) as db:
        query_id = db.create_saved_query(
            name="Local regexp",
            prompt="",
            query_kind="regexp",
            content_scope="papers",
            label_class="Topic",
            label_values=[{"value": "Match", "pattern": "match"}],
        )
        query = db.get_saved_query(query_id)
    resolved = _saved_query_run_options(FormData(), query, config, content_type="papers")
    assert resolved[1] == ()
    with pytest.raises(ValueError, match="Content type must be papers or blogs"):
        _saved_query_run_options(FormData(), query, config, content_type="other")


def test_run_pipeline_search_and_unknown_modes_close_services(config_path, monkeypatch):
    from types import SimpleNamespace
    from paperwatch.web import _run_pipeline

    closed = []

    class FakeStats:
        run_id = 12
        status = "success"

        def as_dict(self):
            return {"papers_found": 1}

    class FakeWatch:
        def __init__(self, *_args, **_kwargs):
            pass

        def fetch_title(self, query):
            assert query == "target"
            return FakeStats(), 44

        def close(self):
            closed.append("watch")

    services = SimpleNamespace(
        db=object(),
        openalex=object(),
        arxiv=object(),
        llm=object(),
        close=lambda: closed.append("services"),
    )
    monkeypatch.setattr("paperwatch.cli.build_services", lambda _config: services)
    monkeypatch.setattr("paperwatch.pipeline.PaperWatch", FakeWatch)
    result = _run_pipeline(config_path, "search", "target")
    assert result["paper_id"] == 44 and result["run_id"] == 12
    assert closed == ["watch", "services"]

    closed.clear()
    with pytest.raises(ValueError, match="Unknown run mode"):
        _run_pipeline(config_path, "invalid")
    assert closed == ["watch", "services"]
