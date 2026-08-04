from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from paperwatch.analysis_jobs import AnalysisBatchSnapshot
from paperwatch.blogs import ArticleItem, BlogFetch
from paperwatch.config import load_config
from paperwatch.db import Database, SCHEMA_VERSION
from paperwatch.pipeline import PaperWatch, RunStats
from paperwatch.query_jobs import QueryBatchSnapshot
from paperwatch.web import RunSnapshot, _render_stored_markdown, create_app


class RunStub:
    def __init__(self, running: bool = False):
        self.running = running

    def snapshot(self):
        return RunSnapshot(running=self.running)

    def start(self, *args, **kwargs):
        return True


class AnalysisStub:
    def __init__(self, running: bool = False):
        self.running = running

    def snapshot(self):
        return AnalysisBatchSnapshot(running=self.running)


class QueryStub:
    def __init__(self, running: bool = False):
        self.running = running

    def snapshot(self):
        return QueryBatchSnapshot(running=self.running)

    def start(self, *args, **kwargs):
        return "batch"


class AuthorClient:
    def close(self):
        return None


class NoPapers:
    def recent_works(self, author_id, *, since, max_results):
        return []

    def close(self):
        return None


class NoLLM:
    provider = "none"
    model = "none"

    def close(self):
        return None


def client(config_path: Path, *, running: bool = False) -> TestClient:
    return TestClient(
        create_app(
            config_path,
            run_controller=RunStub(running),
            analysis_controller=AnalysisStub(running),
            query_controller=QueryStub(running),
            author_client_factory=lambda config: AuthorClient(),
        )
    )


def seed_content(config_path, author_candidate, paper_record):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        paper_id, _ = db.upsert_paper(paper_record)
        blog_id = db.add_author_blog(
            author_candidate.openalex_id,
            "https://example.test/blog/",
            label="Example blog",
        )
        entry_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Markdown website entry",
            content_sha256="entry-096",
            normalized_text="# Heading\n\nBody with [link](https://example.test/reference).",
            rendered_html='<h1>Heading</h1><p>Body with <a href="https://example.test/reference" target="_blank" rel="noreferrer noopener">link</a>.</p>',
            diff_text="",
            change_kind="changed",
            previous_entry_id=None,
            entry_kind="snapshot",
            external_id="entry-096",
            source_url="https://example.test/blog/post",
            published_at="2026-08-02T10:00:00+00:00",
        )
        query_id = db.create_saved_query(
            name="Blog markdown",
            prompt="Summarize.",
            content_scope="blogs",
        )
        run_id = db.create_query_runs(
            batch_id="096",
            query_id=query_id,
            profiles=[("default", "ollama", "test-model")],
            context_window_tokens=2048,
            selection={"content_type": "blogs", "blog_entry_ids": [entry_id]},
        )[0]
        db.finish_query_run(
            run_id,
            matched_count=1,
            input_chars=20,
            output={
                "title": "Result",
                "summary": "## Summary\n\n**Strong** result.",
                "items": [
                    {
                        "blog_entry_id": entry_id,
                        "title": "Markdown website entry",
                        "output": "- First\n- Second",
                    }
                ],
            },
        )
    return paper_id, entry_id, query_id, run_id


def test_query_content_scope_cleanup_and_schema(config_path):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        assert SCHEMA_VERSION == 25
        paper_query = db.create_saved_query(
            name="Paper only",
            prompt="Paper prompt",
            content_scope="papers",
            auto_run_papers=True,
            auto_run_blogs=True,
        )
        blog_query = db.create_saved_query(
            name="Blog only",
            prompt="Blog prompt",
            content_scope="blogs",
            auto_run_papers=True,
            auto_run_blogs=True,
        )
        rows = {int(row["id"]): row for row in db.list_saved_queries()}
        assert rows[paper_query]["auto_run_papers"] == 1
        assert rows[paper_query]["auto_run_blogs"] == 0
        assert rows[blog_query]["auto_run_papers"] == 0
        assert rows[blog_query]["auto_run_blogs"] == 1
        assert [int(row["id"]) for row in db.list_automatic_queries("papers")] == [paper_query]
        assert [int(row["id"]) for row in db.list_automatic_queries("blogs")] == [blog_query]

        db.update_automatic_queries(
            [paper_query, blog_query],
            paper_query_ids=[paper_query, blog_query],
            blog_query_ids=[paper_query, blog_query],
        )
        rows = {int(row["id"]): row for row in db.list_saved_queries()}
        assert rows[paper_query]["auto_run_blogs"] == 0
        assert rows[blog_query]["auto_run_papers"] == 0

        assert db.update_saved_query(
            paper_query,
            name="Paper only",
            prompt="Paper prompt",
            auto_run_blogs=True,
            content_scope="papers",
        )
        assert db.get_saved_query(paper_query)["auto_run_blogs"] == 0
        assert db.update_saved_query(
            blog_query,
            name="Blog only",
            prompt="Blog prompt",
            auto_run_papers=True,
            content_scope="blogs",
        )
        assert db.get_saved_query(blog_query)["auto_run_papers"] == 0
        with pytest.raises(ValueError, match="content scope"):
            db.create_saved_query(name="Bad", prompt="Bad", content_scope="invalid")
        with pytest.raises(ValueError, match="content scope"):
            db.update_saved_query(
                paper_query,
                name="Bad",
                prompt="Bad",
                content_scope="invalid",
            )

        run_ids = db.create_query_runs(
            batch_id="cleanup",
            query_id=paper_query,
            profiles=[("default", "ollama", "test-model")],
            context_window_tokens=1024,
        )
        assert db.delete_query_history() == len(run_ids)
        assert db.get_saved_query(paper_query) is not None
        deleted = db.delete_all_queries()
        assert deleted >= 3
        remaining = db.list_saved_queries()
        assert len(remaining) == 1 and remaining[0]["is_builtin"] == 1


def test_content_author_links_page_jump_query_result_and_controls(
    config_path, author_candidate, paper_record
):
    _paper_id, entry_id, query_id, run_id = seed_content(
        config_path, author_candidate, paper_record
    )
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        blog_id = int(db.get_blog_entry(entry_id)["blog_id"])
        for index in range(25):
            db.save_blog_entry(
                blog_id=blog_id,
                title=f"Extra entry {index}",
                content_sha256=f"extra-{index}",
                normalized_text=f"Extra body {index}",
                diff_text="",
                change_kind="changed",
                previous_entry_id=None,
                entry_kind="snapshot",
                external_id=f"extra-{index}",
                source_url=f"https://example.test/blog/extra-{index}",
            )
    web = client(config_path)
    content = web.get("/?page_size=25")
    assert content.status_code == 200
    assert f'/authors/{author_candidate.openalex_id}/blogs' in content.text
    assert 'max="2"' in content.text and "of 2" in content.text
    assert author_candidate.name in content.text

    blog_page = web.get(f"/blogs/{entry_id}")
    assert "Jump to summary" in blog_page.text
    assert '<details class="panel blog-content-panel" open>' in blog_page.text
    assert 'id="summary"' in blog_page.text
    assert "<h2>Summary</h2>" in blog_page.text
    assert "<strong>Strong</strong>" in blog_page.text
    assert "<ul><li>First</li><li>Second</li></ul>" in blog_page.text

    queries = web.get("/queries")
    assert f'/queries/runs/{run_id}' in queries.text
    assert "Open result" in queries.text
    assert "Strong result" not in queries.text
    result = web.get(f"/queries/runs/{run_id}")
    assert "<h2>Summary</h2>" in result.text

    editor = web.get(f"/queries/{query_id}/edit")
    assert "data-cycle-toggle" in editor.text
    assert 'value="blogs" data-cycle-value' in editor.text

    runs = web.get("/data")
    assert "Delete all queries" in runs.text
    assert "Delete query history" in runs.text
    assert runs.text.index("Authors and blog sources") < runs.text.index("Whole data")
    assert 'data-management-card data-management-wide' in runs.text

    script = Path("src/paperwatch/web_static/app.js").read_text(encoding="utf-8")
    assert "option.selected" in script and "select.selectedIndex = -1" in script
    assert "data-cycle-toggle" in script
    assert "input.value = option.textContent || option.value" in script


def test_query_cleanup_web_actions_and_active_guard(
    config_path, author_candidate, paper_record
):
    _paper_id, _entry_id, query_id, _run_id = seed_content(
        config_path, author_candidate, paper_record
    )
    web = client(config_path)
    response = web.post("/data/delete/query-history", follow_redirects=False)
    assert response.status_code == 303
    with Database(load_config(config_path).paths.database) as db:
        assert db.get_saved_query(query_id) is not None
        assert db.list_query_runs() == []

    response = web.post("/data/delete/queries", follow_redirects=False)
    assert response.status_code == 303
    with Database(load_config(config_path).paths.database) as db:
        assert len(db.list_saved_queries()) == 1

    blocked = client(config_path, running=True)
    for endpoint in ("/data/delete/queries", "/data/delete/query-history"):
        response = blocked.post(endpoint, follow_redirects=False)
        assert response.status_code == 303
        assert "Stop+active+jobs" in response.headers["location"]


def test_markdown_renderer_supports_structure_and_escapes_html():
    rendered = _render_stored_markdown(
        "# Heading\n\n**bold** and *italic* with `code`.\n\n"
        "> quote\n\n1. one\n2. two\n\n```\n<script>alert(1)</script>\n```"
    )
    assert rendered.startswith("<h1>Heading</h1>")
    assert "<strong>bold</strong>" in rendered
    assert "<em>italic</em>" in rendered
    assert "<code>code</code>" in rendered
    assert "<blockquote>quote</blockquote>" in rendered
    assert "<ol><li>one</li><li>two</li></ol>" in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "<script>" not in rendered
    mixed = _render_stored_markdown("- bullet\n1. number\n\n```\nunclosed")
    assert "<ul><li>bullet</li></ul>" in mixed
    assert "<ol><li>number</li></ol>" in mixed
    assert "<pre><code>unclosed</code></pre>" in mixed


class ArxivStub:
    def __init__(self):
        self.calls: list[str] = []

    def by_id(self, arxiv_id):
        self.calls.append(arxiv_id)
        return SimpleNamespace(
            doi=None,
            arxiv_id=arxiv_id,
            title="Imported from RSS",
            abstract="Paper abstract",
            published="2026-08-01",
            updated="2026-08-02",
            abstract_url=f"https://arxiv.org/abs/{arxiv_id}",
            pdf_url=None,
            authors=["Ada Lovelace"],
        )


class EmptyArxivStub:
    def by_id(self, arxiv_id):
        return None


class BrokenArxivStub:
    def by_id(self, arxiv_id):
        raise RuntimeError("broken arxiv")


def test_rss_appends_linked_html_and_imports_only_primary_arxiv_url(
    config_path, author_candidate
):
    feed = """<rss><channel><title>Feed</title>
      <item><guid>paper</guid><title>Paper item</title><link>https://arxiv.org/abs/2607.12345</link><description>Short teaser</description></item>
      <item><guid>post</guid><title>Post item</title><link>https://example.test/post</link><description>See https://arxiv.org/abs/2607.99999</description></item>
    </channel></rss>"""
    page = '<html><head><link rel="alternate" type="application/rss+xml" href="/feed.xml"></head><main>Index</main></html>'

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/blog":
            return httpx.Response(200, headers={"content-type": "text/html"}, text=page)
        if request.url.path == "/feed.xml":
            return httpx.Response(200, headers={"content-type": "application/rss+xml"}, text=feed)
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text='<main><h1>Full article</h1><p>Detailed body with <a href="https://example.test/reference">reference</a>.</p></main>',
        )

    transport_client = httpx.Client(transport=httpx.MockTransport(handler))
    arxiv = ArxivStub()
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        db.add_author_blog(author_candidate.openalex_id, "https://example.test/blog")
        app = PaperWatch(
            config,
            db=db,
            openalex=NoPapers(),
            arxiv=arxiv,
            llm=NoLLM(),
            pdf_client=transport_client,
        )
        app._process_pdf = lambda *args, **kwargs: None
        stats = app.run(today=date(2026, 8, 2))
        assert stats.status == "success"
        rss_entries = [row for row in db.list_blog_entries() if row["entry_kind"] == "rss"]
        assert len(rss_entries) == 2
        assert all("Website content:" in row["normalized_text"] for row in rss_entries)
        assert all("Detailed body" in row["normalized_text"] for row in rss_entries)
        assert all('href="https://example.test/reference"' in row["rendered_html"] for row in rss_entries)
        papers, total = db.search_papers(query="Imported from RSS")
        assert total == 1 and papers[0]["arxiv_id"] == "2607.12345"
        assert arxiv.calls == ["2607.12345"]
    transport_client.close()


def test_unchanged_feed_enriches_existing_teaser_in_place(
    config_path, author_candidate
):
    config = load_config(config_path)
    client_transport = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(404))
    )
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        blog_id = db.add_author_blog(
            author_candidate.openalex_id,
            "https://example.test/blog",
        )
        entry_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Teaser",
            content_sha256="old",
            normalized_text="Title: Teaser\nURL: https://example.test/post\nShort teaser",
            rendered_html="<p>Short teaser</p>",
            diff_text="",
            change_kind="changed",
            previous_entry_id=None,
            entry_kind="rss",
            external_id="teaser",
            source_url="https://example.test/post",
        )
        article = ArticleItem(
            external_id="article",
            title="Full post",
            url="https://example.test/post",
            text="Title: Full post\nURL: https://example.test/post\n\nFull body",
            sha256="article-hash",
            rendered_html='<p>Full body with <a href="https://example.test/ref">reference</a>.</p>',
        )
        app = PaperWatch(
            config,
            db=db,
            openalex=NoPapers(),
            arxiv=None,
            llm=NoLLM(),
            pdf_client=client_transport,
        )
        stats = RunStats(run_id=db.start_run())
        blog = db.author_blogs(author_candidate.openalex_id)[0]
        app._apply_blog_fetch(
            stats,
            blog,
            BlogFetch(
                method="rss",
                title="Feed",
                status_code=304,
                article_items=(article,),
                source_mode="auto",
            ),
            None,
        )
        enriched = db.get_blog_entry(entry_id)
        assert enriched is not None
        assert "Website content:" in enriched["normalized_text"]
        assert "Full body" in enriched["normalized_text"]
        assert 'href="https://example.test/ref"' in enriched["rendered_html"]
        assert len([row for row in db.list_blog_entries() if row["entry_kind"] == "rss"]) == 1
        assert stats.blog_updates_found == 0
    client_transport.close()


def test_arxiv_feed_import_skip_and_error_branches(config_path):
    config = load_config(config_path)
    with Database(config.paths.database) as db:
        app = PaperWatch(
            config,
            db=db,
            openalex=NoPapers(),
            arxiv=EmptyArxivStub(),
            llm=NoLLM(),
            pdf_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(404))),
        )
        stats = RunStats(run_id=db.start_run())
        app._import_arxiv_feed_item(stats, None)
        app._import_arxiv_feed_item(stats, "https://example.test/2607.1")
        app._import_arxiv_feed_item(stats, "https://arxiv.org/not-an-id")
        app._import_arxiv_feed_item(stats, "https://arxiv.org/abs/2607.11111")
        assert stats.errors_count == 0
        app.arxiv = BrokenArxivStub()
        app._import_arxiv_feed_item(stats, "https://arxiv.org/abs/2607.22222")
        assert stats.errors_count == 1
        app.pdf_client.close()
