from __future__ import annotations

import hashlib
import threading
from dataclasses import replace
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from fastapi.testclient import TestClient

import paperwatch.pipeline as pipeline_module
from paperwatch.analysis_jobs import AnalysisBatchSnapshot
from paperwatch.blogs import ArticleItem, BlogFetch, FeedItem, parse_feed
from paperwatch.config import load_config
from paperwatch.db import Database, SCHEMA_VERSION
from paperwatch.models import PaperRecord
from paperwatch.pipeline import PaperWatch, RunStats
from paperwatch.query_jobs import QueryBatchSnapshot
from paperwatch.web import RunSnapshot, _website_view, create_app


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


class RunStub:
    def snapshot(self):
        return RunSnapshot()

    def start(self, *args, **kwargs):
        return True


class AnalysisStub:
    def snapshot(self):
        return AnalysisBatchSnapshot()


class QueryStub:
    def snapshot(self):
        return QueryBatchSnapshot()

    def start(self, *args, **kwargs):
        return "batch"


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


def test_arxiv_feed_urls_are_https_and_clickable():
    feed = """<rss><channel><title>arXiv feed</title><item>
      <title>arXiv</title><link>http://arxiv.org/abs/1606.02275</link>
      <description>arXiv</description></item></channel></rss>"""
    _title, items = parse_feed(feed, "https://example.test/feed", "Fallback")
    assert len(items) == 1
    assert items[0].url == "https://arxiv.org/abs/1606.02275"
    assert 'href="https://arxiv.org/abs/1606.02275"' in (items[0].rendered_html or "")
    assert "Open source" in (items[0].rendered_html or "")


def test_arxiv_rss_fallback_links_entry_to_paper_and_query_result(
    config_path, author_candidate
):
    config = load_config(config_path)
    article_url = "https://arxiv.org/abs/1606.02275"
    article_text = (
        "Title: A Useful arXiv Paper\n"
        f"URL: {article_url}\n"
        "This is the abstract extracted from the linked arXiv page."
    )
    article = ArticleItem(
        external_id="article-1606",
        title="A Useful arXiv Paper",
        url=article_url,
        text=article_text,
        sha256=hashlib.sha256(article_text.encode()).hexdigest(),
        rendered_html="<h1>A Useful arXiv Paper</h1><p>This is the abstract.</p>",
    )
    item_text = f"Title: arXiv\nURL: {article_url}\narXiv"
    item = FeedItem(
        external_id="feed-1606",
        title="arXiv",
        url=article_url,
        published_at="2016-06-08T00:00:00+00:00",
        text=item_text,
        sha256=hashlib.sha256(item_text.encode()).hexdigest(),
        rendered_html=f'<p><a href="{article_url}">Open source</a></p><p>arXiv</p>',
    )
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(404))
    )
    with Database(config.paths.database) as db:
        db.add_author(author_candidate)
        db.add_author_blog(author_candidate.openalex_id, "https://example.test/feed")
        blog = db.author_blogs(author_candidate.openalex_id)[0]
        app = PaperWatch(
            config,
            db=db,
            openalex=NoPapers(),
            arxiv=None,
            llm=NoLLM(),
            pdf_client=client,
        )
        stats = RunStats(run_id=db.start_run())
        app._apply_blog_fetch(
            stats,
            blog,
            BlogFetch(
                method="rss",
                title="arXiv feed",
                status_code=200,
                feed_items=(item,),
                article_items=(article,),
                source_mode="auto",
                feed_url="https://example.test/feed.xml",
            ),
            None,
        )
        assert stats.errors_count == 0
        papers, total = db.search_papers(query="Useful arXiv")
        assert total == 1
        paper_id = int(papers[0]["id"])
        rss_entry = next(row for row in db.list_blog_entries() if row["entry_kind"] == "rss")
        entry_id = int(rss_entry["id"])
        assert rss_entry["linked_paper_id"] == paper_id
        query_id = db.create_saved_query(name="One blog", prompt="Summarize")
        run_id = db.create_query_runs(
            batch_id="linked-source",
            query_id=query_id,
            profiles=[("default", "ollama", "model")],
            context_window_tokens=2048,
            selection={
                "content_type": "blogs",
                "source_scope": "blog",
                "blog_entry_ids": [entry_id],
                "labels": ["Blog entry: arXiv"],
            },
        )[0]
        db.finish_query_run(
            run_id,
            matched_count=1,
            input_chars=10,
            output={"title": "Result", "summary": "Done", "items": []},
        )
        db.record_error(
            stats.run_id,
            stage="test",
            message="must stay off content page",
        )

    web = web_client(config_path)
    detail = web.get(f"/blogs/{entry_id}")
    assert f'href="/papers/{paper_id}"' in detail.text
    assert "Open paper in Paperwatch" in detail.text
    query_detail = web.get(f"/queries/runs/{run_id}")
    assert f'<a href="/blogs/{entry_id}">Blog entry: arXiv</a>' in query_detail.text
    home = web.get("/?content=blogs")
    assert "must stay off content page" not in home.text
    soup = BeautifulSoup(home.text, "html.parser")
    source = soup.select_one("tr.content-row-blog td.paper-date-cell small a")
    assert source is not None
    assert source.get_text(strip=True) == "arxiv.org"
    assert source.get("href") == "https://arxiv.org/"
    client.close()


def test_schema_21_backfills_arxiv_blog_paper_links(
    tmp_path, author_candidate, paper_record
):
    path = tmp_path / "schema-21.sqlite"
    with Database(path) as db:
        db.add_author(author_candidate)
        paper = replace(paper_record, arxiv_id="1606.02275")
        paper_id, _ = db.upsert_paper(paper)
        blog_id = db.add_author_blog(author_candidate.openalex_id, "https://example.test/feed")
        entry_id = db.save_blog_entry(
            blog_id=blog_id,
            title="arXiv",
            content_sha256="hash",
            normalized_text="Title: arXiv\nURL: http://arxiv.org/abs/1606.02275\narXiv",
            diff_text="",
            change_kind="changed",
            previous_entry_id=None,
            entry_kind="rss",
            external_id="old-arxiv",
            source_url="http://arxiv.org/abs/1606.02275v2",
        )
        db._conn.execute("UPDATE blog_entries SET linked_paper_id=NULL WHERE id=?", (entry_id,))
        db._conn.execute("PRAGMA user_version=20")
        db._conn.commit()
    with Database(path) as db:
        assert SCHEMA_VERSION == 26
        assert db.get_blog_entry(entry_id)["linked_paper_id"] == paper_id


def test_blog_fetch_worker_pool_is_bounded(config_path, monkeypatch):
    config = load_config(config_path)
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(404))
    )
    lock = threading.Lock()
    release = threading.Event()
    active = 0
    peak = 0

    def fake_fetch(_client, row):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if peak == pipeline_module.MAX_BLOG_FETCH_WORKERS:
                release.set()
        assert release.wait(timeout=3)
        with lock:
            active -= 1
        return BlogFetch(
            method="snapshot",
            title=str(row["id"]),
            status_code=200,
            snapshot_text="body",
            snapshot_sha256="hash",
        )

    monkeypatch.setattr(pipeline_module, "fetch_blog", fake_fetch)
    with Database(config.paths.database) as db:
        app = PaperWatch(
            config,
            db=db,
            openalex=NoPapers(),
            arxiv=None,
            llm=NoLLM(),
            pdf_client=client,
        )
        rows = [
            {"id": index, "url": f"https://example.test/{index}"}
            for index in range(pipeline_module.MAX_BLOG_FETCH_WORKERS + 4)
        ]
        results = app._fetch_blog_batch(rows)
    assert len(results) == len(rows)
    assert peak == pipeline_module.MAX_BLOG_FETCH_WORKERS
    assert all(error is None for _row, _fetched, error in results)
    client.close()


def test_dropdown_and_delete_layout_assets(config_path):
    web = web_client(config_path)
    script = web.get("/static/app.js").text
    assert "enabledOptions.length <= 5" in script
    assert "form.requestSubmit()" in script
    assert "search-select-cycle" in script
    runs = web.get("/data")
    assert "delete-record-grid" in runs.text
    assert "👥" in runs.text and "📄" in runs.text and "📰" in runs.text
    assert "Downloaded content" in runs.text


def test_small_helper_and_paper_query_result_link(
    config_path, paper_record
):
    assert _website_view("not a URL") == ("not a URL", "not a URL")
    assert PaperWatch._arxiv_fallback_title("1606.02275", "arXiv", None) == "arXiv 1606.02275"
    assert PaperWatch._arxiv_fallback_abstract(None) is None

    config = load_config(config_path)
    with Database(config.paths.database) as db:
        paper_id, _ = db.upsert_paper(paper_record)
        query_id = db.create_saved_query(name="One paper", prompt="Summarize")
        run_id = db.create_query_runs(
            batch_id="paper-source-link",
            query_id=query_id,
            profiles=[("default", "ollama", "model")],
            context_window_tokens=2048,
            selection={
                "content_type": "papers",
                "source_scope": "abstract",
                "paper_ids": [paper_id, "bad", 999999],
            },
        )[0]
        db.finish_query_run(
            run_id,
            matched_count=1,
            input_chars=10,
            output={"title": "Result", "summary": "Done", "items": []},
        )
    detail = web_client(config_path).get(f"/queries/runs/{run_id}")
    assert f'<a href="/papers/{paper_id}">Paper: {paper_record.title}</a>' in detail.text
