from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from paperwatch.analysis_jobs import AnalysisBatchSnapshot
from paperwatch.config import load_config
from paperwatch.db import (
    BLOG_STUB_DIFF_MAX_WORDS,
    Database,
    _blog_entry_is_stub,
    _blog_entry_word_count,
)
from paperwatch.pipeline import PaperWatch
from paperwatch.query_jobs import QueryBatchSnapshot
from paperwatch.web import RunSnapshot, create_app


class RunStub:
    def snapshot(self):
        return RunSnapshot()


class AnalysisStub:
    def snapshot(self):
        return AnalysisBatchSnapshot()


class QueryStub:
    def snapshot(self):
        return QueryBatchSnapshot()


class AuthorClient:
    def close(self):
        return None


def web_client(config_path: Path) -> TestClient:
    return TestClient(
        create_app(
            config_path,
            run_controller=RunStub(),
            analysis_controller=AnalysisStub(),
            query_controller=QueryStub(),
            author_client_factory=lambda config: AuthorClient(),
        )
    )


def test_content_extent_stub_button_and_colored_collapsed_diff(
    config_path, author_candidate, paper_record
):
    client = web_client(config_path)
    config = load_config(config_path)
    pdf = config.paths.pdf_dir / "three-pages.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=72, height=72)
    with pdf.open("wb") as handle:
        writer.write(handle)

    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        paper_id, _ = db.upsert_paper(paper_record)
        db.set_local_pdf(paper_id, str(pdf))
        # Keep this rendering test independent of pypdf's handling of blank pages.
        db.set_paper_pdf_page_count(paper_id, 3)
        blog_id = db.add_author_blog(
            author_candidate.openalex_id,
            "https://example.test/words",
            label="Words",
        )
        baseline = db.save_blog_entry(
            blog_id=blog_id,
            title="Baseline",
            content_sha256="words-baseline",
            normalized_text="one two three four five",
            diff_text="",
            change_kind="changed",
            previous_entry_id=None,
            source_url="https://example.test/words/baseline",
        )
        added = " ".join(f"added{index}" for index in range(BLOG_STUB_DIFF_MAX_WORDS + 2))
        updated_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Large update",
            content_sha256="words-large-update",
            normalized_text="old text " + added,
            diff_text=f"-old text\n+{added}",
            change_kind="changed",
            previous_entry_id=baseline,
            source_url="https://example.test/words/large",
        )
        stub_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Tiny update",
            content_sha256="words-tiny-update",
            normalized_text="old text tiny addition",
            diff_text="-old text\n+tiny addition",
            change_kind="changed",
            previous_entry_id=updated_id,
            source_url="https://example.test/words/tiny",
        )

    page = client.get("/")
    assert page.status_code == 200
    assert "3p" in page.text
    assert f"{BLOG_STUB_DIFF_MAX_WORDS + 2}w" in page.text
    assert "Tiny update" not in page.text
    assert 'data-filter-name="stub"' in page.text
    assert 'data-filter-state="no"' in page.text
    assert 'data-filter-values="no|yes|all"' in page.text
    assert "display stubs" not in page.text.casefold()
    assert page.text.index('data-filter-name="starred"') < page.text.index(
        'data-filter-name="stub"'
    ) < page.text.index('data-filter-name="content"')

    shown = client.get("/?content=blogs&stub=yes")
    assert "Tiny update" in shown.text
    assert "Large update" not in shown.text
    assert "2w" in shown.text
    either = client.get("/?content=blogs&stub=all")
    assert "Tiny update" in either.text and "Large update" in either.text

    detail = client.get(f"/blogs/{stub_id}")
    assert detail.status_code == 200
    assert '<details class="panel blog-diff-panel">' in detail.text
    assert '<details class="panel blog-diff-panel" open' not in detail.text
    assert "blog-diff-added" in detail.text
    css = (Path(__file__).parents[1] / "src" / "paperwatch" / "web_static" / "app.css").read_text()
    assert ".blog-diff-added{color:var(--good)" in css
    assert ".blog-diff-removed{color:var(--danger)" in css
    assert ".content-extent-heading,.content-extent-cell" in css
    assert "blog-diff-removed" in detail.text


def test_stub_state_is_auto_assigned_filterable_and_manually_editable(
    config_path, author_candidate
):
    client = web_client(config_path)
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        blog_id = db.add_author_blog(author_candidate.openalex_id, "https://example.test/stubs")
        auto_stub_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Metadata stub",
            content_sha256="manual-stub",
            normalized_text="Title: Metadata stub\nURL: https://example.test/stubs/item",
            diff_text="",
            change_kind="changed",
            previous_entry_id=None,
            source_url="https://example.test/stubs/item",
        )
        full_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Full article",
            content_sha256="manual-full",
            normalized_text=" ".join(f"word{i}" for i in range(150)),
            diff_text="",
            change_kind="changed",
            previous_entry_id=None,
            source_url="https://example.test/stubs/full",
        )
        assert [row["id"] for row in db.list_blog_entries(stub=True)] == [auto_stub_id]
        assert [row["id"] for row in db.list_blog_entries(stub=False)] == [full_id]

    stub_page = client.get("/?content=blogs&stub=yes")
    assert "Metadata stub" in stub_page.text
    assert 'class="icon-button stub-state-button is-stub"' in stub_page.text
    assert 'data-filter-state-value="no">F</span>' in stub_page.text
    assert 'data-filter-state-value="yes">S</span>' in stub_page.text
    assert 'data-filter-state-value="all">/</span>' in stub_page.text
    assert 'name="stub_state" value="stub"' in stub_page.text
    assert 'aria-label="Edit stub state">/</button>' in stub_page.text
    changed = client.post(
        f"/content/blog/{auto_stub_id}/stub",
        data={"stub_state": "full"},
        headers={"X-Paperwatch-Async": "1"},
    )
    assert changed.status_code == 200
    assert changed.json()["stub_state"] == "full"
    assert changed.json()["is_stub"] is False
    assert "Metadata stub" in client.get("/?content=blogs").text
    assert "Metadata stub" not in client.get("/?content=blogs&stub=yes").text

    detail = client.get(f"/blogs/{auto_stub_id}")
    assert "Stub: Full" in detail.text
    reset = client.post(
        f"/content/blog/{auto_stub_id}/stub",
        data={"stub_state": "auto"},
        headers={"X-Paperwatch-Async": "1"},
    )
    assert reset.json()["is_stub"] is True
    auto_page = client.get("/?content=blogs&stub=yes")
    assert "Metadata stub" in auto_page.text
    assert 'name="stub_state" value="stub"' in auto_page.text
    assert 'aria-label="Edit stub state">/</button>' in auto_page.text

    manual_stub = client.post(
        f"/content/blog/{auto_stub_id}/stub",
        data={"stub_state": "stub"},
        headers={"X-Paperwatch-Async": "1"},
    )
    assert manual_stub.json()["stub_state"] == "stub"
    manual_stub_page = client.get("/?content=blogs&stub=yes")
    assert 'name="stub_state" value="full"' in manual_stub_page.text
    assert 'aria-label="Edit stub state">S</button>' in manual_stub_page.text

    manual_full = client.post(
        f"/content/blog/{auto_stub_id}/stub",
        data={"stub_state": "full"},
        headers={"X-Paperwatch-Async": "1"},
    )
    assert manual_full.json()["stub_state"] == "full"
    manual_full_page = client.get("/?content=blogs")
    assert 'name="stub_state" value="auto"' in manual_full_page.text
    assert 'aria-label="Edit stub state">F</button>' in manual_full_page.text



def test_blog_word_count_and_stub_diff_helpers():
    assert _blog_entry_word_count("one two three", "", None) == 3
    assert _blog_entry_word_count("ignored full text", "-old words\n+one two three", 1) == 3
    assert _blog_entry_is_stub(
        "A full article body that is meaningful.",
        "Updated",
        "https://example.test/post",
        None,
        "-old wording\n+new wording",
        1,
    ) == 1
    large = "+" + " ".join(
        f"word{index}" for index in range(BLOG_STUB_DIFF_MAX_WORDS + 1)
    )
    assert _blog_entry_is_stub(
        "A full article body that is meaningful.",
        "Updated",
        "https://example.test/post",
        None,
        large,
        1,
    ) == 0
    # A first version is not a diff-stub merely because import code supplied a + line.
    assert _blog_entry_is_stub(
        "A full article body that is meaningful.",
        "New",
        "https://example.test/post",
        None,
        "+new wording",
        None,
    ) == 0


def test_shared_author_panel_has_local_papers_and_blogs(
    config_path, author_candidate, paper_record
):
    client = web_client(config_path)
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        paper_id, _ = db.upsert_paper(paper_record)
        blog_id = db.add_author_blog(author_candidate.openalex_id, "https://example.test/blog")
        entry_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Author panel entry",
            content_sha256="author-panel-entry",
            normalized_text="A substantive blog article for the shared author panel.",
            diff_text="",
            change_kind="changed",
            previous_entry_id=None,
            source_url="https://example.test/blog/entry",
        )

    for url in (f"/papers/{paper_id}", f"/blogs/{entry_id}"):
        response = client.get(url)
        assert response.status_code == 200
        assert 'class="panel author-picker-panel"' in response.text
        assert 'id="author-local-papers"' in response.text
        assert 'id="author-local-blogs"' in response.text
        assert "/?content=papers&amp;author=A123" in response.text
        assert "/?content=blogs&amp;author=A123" in response.text


def test_fresh_web_database_installs_defaults_and_pdf_setting_persists(config_path):
    config = load_config(config_path)
    assert not config.paths.database.exists()
    client = web_client(config_path)
    with Database(config.paths.database) as db:
        optional = [row for row in db.list_saved_queries() if not bool(row["is_builtin"])]
        assert len(optional) == 30
        assert db.download_pdfs_enabled() is True

    data = client.get("/data")
    assert data.status_code == 200
    assert data.text.index("Delete stored records") < data.text.index("Defaults and fetching")
    assert "Download PDFs automatically" in data.text
    assert "Master enabled · click to disable" in data.text

    disabled = client.post(
        "/data/settings/download-pdfs",
        data={"enabled": "0"},
        follow_redirects=False,
    )
    assert disabled.status_code == 303
    with Database(config.paths.database) as db:
        assert db.download_pdfs_enabled() is False

    data = client.get("/data")
    assert "Master disabled · click to enable" in data.text
    enabled = client.post(
        "/data/settings/download-pdfs",
        data={"enabled": "1"},
        follow_redirects=False,
    )
    assert enabled.status_code == 303
    with Database(config.paths.database) as db:
        assert db.download_pdfs_enabled() is True


def test_automatic_pipeline_obeys_persistent_pdf_download_setting(
    app_config, author_candidate, paper_record
):
    class OpenAlex:
        def recent_works(self, author_id, *, since, max_results):
            return [paper_record]

        def close(self):
            return None

    class NoLLM:
        provider = "none"

        def close(self):
            return None

    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(200, content=b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF\n")

    with Database(app_config.paths.database) as db:
        db.add_author(author_candidate)
        db.set_download_pdfs_enabled(False)
        watch = PaperWatch(
            app_config,
            db=db,
            openalex=OpenAlex(),
            arxiv=None,
            llm=NoLLM(),
            pdf_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        stats = watch.run(today=date(2026, 7, 24), analyze_discovered_papers=False)
        assert stats.pdfs_downloaded == 0
        assert requests == []
        assert not db.list_papers()[0]["local_pdf_path"]
