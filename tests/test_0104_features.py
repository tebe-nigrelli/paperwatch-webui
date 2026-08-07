from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from paperwatch.analysis_jobs import AnalysisBatchSnapshot
from paperwatch.config import load_config
from paperwatch.db import Database
from paperwatch.pipeline import PaperWatch
from paperwatch.query_jobs import QueryBatchSnapshot
from paperwatch.web import RunSnapshot, _format_word_count, create_app


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


def test_blog_word_counts_use_kw_and_keep_small_counts_as_words(
    config_path, author_candidate
):
    assert _format_word_count(0) == "0w"
    assert _format_word_count(999) == "999w"
    assert _format_word_count(1000) == "1Kw"
    assert _format_word_count(1600) == "1.6Kw"

    client = web_client(config_path)
    config = load_config(config_path)
    body = " ".join(f"word{i}" for i in range(1600))
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        blog_id = db.add_author_blog(
            author_candidate.openalex_id, "https://example.test/long-blog"
        )
        db.save_blog_entry(
            blog_id=blog_id,
            title="Long blog",
            content_sha256="long-blog",
            normalized_text=body,
            diff_text="",
            change_kind="changed",
            previous_entry_id=None,
            source_url="https://example.test/long-blog/post",
        )

    page = client.get("/?content=blogs")
    assert page.status_code == 200
    assert "1.6Kw" in page.text
    assert "1600w" not in page.text


def test_appended_blog_fetch_counts_full_accumulated_text(
    config_path, author_candidate
):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        blog_id = db.add_author_blog(
            author_candidate.openalex_id, "https://example.test/appended-blog"
        )
        baseline_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Baseline",
            content_sha256="baseline",
            normalized_text="old teaser",
            diff_text="",
            change_kind="baseline",
            previous_entry_id=None,
            source_url="https://example.test/appended-blog/post",
        )
        article_words = " ".join(f"article{i}" for i in range(1500))
        accumulated = f"feed teaser\n\nWebsite content:\n{article_words}"
        entry_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Fetched article",
            content_sha256="fetched",
            normalized_text=accumulated,
            diff_text="--- previous\n+++ current\n-old\n+tiny change",
            change_kind="changed",
            previous_entry_id=baseline_id,
            source_url="https://example.test/appended-blog/post",
        )
        row = next(
            row
            for row in db.list_blog_entries(table_view=True, stub=None)
            if int(row["id"]) == entry_id
        )
        assert int(row["content_extent"]) == len(accumulated.split())
        assert int(row["content_extent"]) > 1500

        imported = f"metadata\n\nImported page content:\n{article_words}"
        imported_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Enriched stub",
            content_sha256="imported",
            normalized_text=imported,
            diff_text="--- previous\n+++ current\n-old\n+small addition",
            change_kind="changed",
            previous_entry_id=baseline_id,
            source_url="https://example.test/appended-blog/other",
        )
        imported_row = next(
            row
            for row in db.list_blog_entries(table_view=True, stub=None)
            if int(row["id"]) == imported_id
        )
        assert int(imported_row["content_extent"]) == len(imported.split())

        normal_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Normal revision",
            content_sha256="normal",
            normalized_text="this is the complete revised article body",
            diff_text="--- previous\n+++ current\n+only these three",
            change_kind="changed",
            previous_entry_id=baseline_id,
            source_url="https://example.test/appended-blog/normal",
        )
        normal_row = next(
            row
            for row in db.list_blog_entries(table_view=True, stub=None)
            if int(row["id"]) == normal_id
        )
        assert int(normal_row["content_extent"]) == 3


def test_linked_sources_table_has_pause_resume_and_remove(
    config_path, author_candidate
):
    client = web_client(config_path)
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        blog_id = db.add_author_blog(
            author_candidate.openalex_id,
            "https://example.test/source",
            label="Source",
        )

    page = client.get(f"/authors/{author_candidate.openalex_id}/blogs")
    assert page.status_code == 200
    assert 'class="author-blog-table linked-sources-table"' in page.text
    assert 'class="linked-sources-columns"' in page.text
    assert '<th class="linked-source-icon-heading">Open</th>' in page.text
    assert '<th>Tags</th>' in page.text
    assert 'class="source-tag-cell"' in page.text
    assert 'class="blog-source-notes-row' in page.text
    assert 'colspan="10"' in page.text
    assert '>Edit source</summary>' in page.text
    assert f'/blogs/{blog_id}/pause' in page.text
    assert ">⏸<" in page.text
    assert ">×</button>" in page.text

    css = client.get("/static/app.css").text
    assert ".icon-button{display:inline-grid;place-items:center" in css
    assert ".linked-sources-table col.linked-source-col-tags{width:15%}" in css
    assert ".blog-source-notes-row .row-notes-editor{width:100%;min-width:0;max-width:none" in css

    paused = client.post(
        f"/authors/{author_candidate.openalex_id}/blogs/{blog_id}/pause",
        follow_redirects=False,
    )
    assert paused.status_code == 303
    with Database(config.paths.database) as db:
        assert db.author_blogs(author_candidate.openalex_id)[0]["enabled"] == 0

    page = client.get(f"/authors/{author_candidate.openalex_id}/blogs")
    assert f'/blogs/{blog_id}/resume' in page.text
    assert ">▶<" in page.text
    assert "paused" in page.text

    resumed = client.post(
        f"/authors/{author_candidate.openalex_id}/blogs/{blog_id}/resume",
        follow_redirects=False,
    )
    assert resumed.status_code == 303
    with Database(config.paths.database) as db:
        assert db.author_blogs(author_candidate.openalex_id)[0]["enabled"] == 1

    removed = client.post(
        f"/authors/{author_candidate.openalex_id}/blogs/{blog_id}/remove",
        follow_redirects=False,
    )
    assert removed.status_code == 303
    with Database(config.paths.database) as db:
        assert db.author_blogs(author_candidate.openalex_id) == []


def test_author_pdf_toggle_is_persistent_filterable_and_master_gated(
    config_path, author_candidate
):
    client = web_client(config_path)
    config = load_config(config_path)
    second = replace(author_candidate, openalex_id="A456", name="Grace Hopper")
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        db.add_author(second)
        db.set_author_download_papers(author_candidate.openalex_id, False)

    disabled = client.get("/authors?download_papers=no")
    assert disabled.status_code == 200
    assert author_candidate.name in disabled.text
    assert second.name not in disabled.text
    assert 'data-filter-name="download_papers"' in disabled.text
    assert 'data-filter-state="no"' in disabled.text

    enabled = client.get("/authors?download_papers=yes")
    assert second.name in enabled.text
    assert author_candidate.name not in enabled.text

    toggled = client.post(
        f"/authors/{author_candidate.openalex_id}/download-papers",
        data={"enabled": "1"},
        follow_redirects=False,
    )
    assert toggled.status_code == 303
    with Database(config.paths.database) as db:
        assert db.get_author(author_candidate.openalex_id)["download_papers"] == 1

    data = client.get("/data")
    assert "master PDF-download switch" in data.text
    assert "Author-level PDF toggles" in data.text


def test_pipeline_obeys_per_author_pdf_preference_under_master(
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

    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(
            200,
            content=b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF\n",
            headers={"content-type": "application/pdf"},
        )

    with Database(app_config.paths.database) as db:
        db.add_author(author_candidate)
        db.set_download_pdfs_enabled(True)
        db.set_author_download_papers(author_candidate.openalex_id, False)
        watch = PaperWatch(
            app_config,
            db=db,
            openalex=OpenAlex(),
            arxiv=None,
            llm=NoLLM(),
            pdf_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        stats = watch.run(today=date(2026, 8, 7), analyze_discovered_papers=False)
        assert stats.pdfs_downloaded == 0
        assert requests == []
        assert not db.list_papers()[0]["local_pdf_path"]


def test_builtin_query_without_delete_aligns_edit_to_top(config_path):
    client = web_client(config_path)
    page = client.get("/queries")
    assert page.status_code == 200
    assert "saved-query-actions-no-delete" in page.text
    css = (Path(__file__).parents[1] / "src" / "paperwatch" / "web_static" / "app.css").read_text()
    assert ".saved-query-actions-no-delete{vertical-align:top}" in css
