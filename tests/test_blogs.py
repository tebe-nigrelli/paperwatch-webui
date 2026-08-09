from __future__ import annotations

import io
import hashlib
import sqlite3
from datetime import date
from pathlib import Path

import httpx
import pytest

from paperwatch import blogs
from paperwatch.blog_import import import_blog_links, import_blog_links_path
from paperwatch.db import Database
from paperwatch.models import AuthorCandidate
from paperwatch.pipeline import PaperWatch


class NoPapers:
    def recent_works(self, author_id, *, since, max_results):
        return []

    def close(self):
        pass


class NoLLM:
    provider = "none"
    model = "none"

    def close(self):
        pass


def test_normalize_blog_url_and_validation():
    assert blogs.normalize_blog_url(" HTTPS://Example.COM:443/blog#part ") == "https://example.com/blog"
    assert blogs.normalize_blog_url("http://example.com:8080") == "http://example.com:8080/"
    for value in ("", "ftp://example.com", "http://localhost/x", "http://127.0.0.1/x"):
        with pytest.raises(ValueError):
            blogs.normalize_blog_url(value)
    with pytest.raises(ValueError):
        blogs.normalize_blog_url("https://" + "a" * 2001)


def test_html_date_prefers_structured_publication_metadata():
    assert blogs._html_first_date(
        """<html><head>
        <meta property="article:published_time" content="2026-06-30T08:15:00Z">
        </head><body><p>Copyright 2027</p></body></html>"""
    ) == "2026-06-30"
    assert blogs._html_first_date(
        """<html><body><time datetime="2020-07-24">Jul 24, 2020</time>
        <p>Updated footer 2026</p></body></html>"""
    ) == "2020-07-24"
    assert blogs._html_first_date(
        """<script type="application/ld+json">
        {"@type":"BlogPosting","datePublished":"2024-05-09"}
        </script><main>Article text</main>"""
    ) == "2024-05-09"


def test_extract_fetch_and_diff_branches(monkeypatch):
    title, text = blogs.extract_blog_text(
        "<html><head><title> Site </title></head><body><nav>Noise</nav><main><h1>Post</h1><p>Hello  world</p><p>Hello world</p></main></body></html>",
        "Fallback",
    )
    assert title == "Site"
    assert text == "Post\nHello world"
    fallback_title, _ = blogs.extract_blog_text("<article>Text</article>", "Fallback")
    assert fallback_title == "Fallback"
    with pytest.raises(ValueError):
        blogs.extract_blog_text("<html><script>x</script></html>", "Fallback")

    row = {
        "url": "https://example.com/blog",
        "title": "Blog",
        "etag": "one",
        "last_modified": "yesterday",
        "feed_discovery_checked": 1,
    }
    seen = {}

    def handler(request):
        seen.update(request.headers)
        return httpx.Response(
            200,
            headers={"content-type": "text/html", "etag": "two", "last-modified": "today"},
            text="<title>Blog title</title><main>New text</main>",
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        snapshot = blogs.fetch_blog(client, row)
    assert snapshot.title == "Blog title"
    assert snapshot.method == "snapshot"
    assert snapshot.snapshot_text == "New text"
    assert seen["if-none-match"] == "one"
    assert seen["if-modified-since"] == "yesterday"

    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(304))) as client:
        unchanged = blogs.fetch_blog(client, row)
        assert unchanged.not_modified and unchanged.method == "snapshot"
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"x"))) as client:
        with pytest.raises(ValueError, match="Unsupported"):
            blogs.fetch_blog(client, row)
    monkeypatch.setattr(blogs, "MAX_RESPONSE_BYTES", 2)
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, headers={"content-type": "text/plain"}, content=b"long"))) as client:
        with pytest.raises(ValueError, match="exceeds"):
            blogs.fetch_blog(client, row)

    assert blogs.blog_diff(None, "a") == ("baseline", "")
    kind, diff = blogs.blog_diff("a", "b")
    assert kind == "changed" and "-a" in diff and "+b" in diff
    kind, diff = blogs.blog_diff(
        "header\nunchanged one\nold value\nunchanged two\nfooter",
        "header\nunchanged one\nnew value\nunchanged two\nfooter",
    )
    assert kind == "changed"
    assert "-old value" in diff and "+new value" in diff
    assert "unchanged one" not in diff and "unchanged two" not in diff
    assert "--- previous" not in diff and "+++ current" not in diff
    assert blogs.compact_blog_diff(
        "--- previous\n+++ current\n@@ -1,3 +1,3 @@\n context\n-old\n+new"
    ) == "@@ -1,3 +1,3 @@\n-old\n+new"
    kind, _ = blogs.blog_diff("\n".join(f"old {i}" for i in range(30)), "\n".join(f"new {i}" for i in range(30)))
    assert kind == "reset"
    monkeypatch.setattr(blogs, "MAX_DIFF_CHARS", 10)
    kind, diff = blogs.blog_diff("a\nb\nc", "d\ne\nf")
    assert kind == "reset" and "truncated" in diff


def test_rss_atom_parsing_discovery_and_fetch_fallback():
    rss = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>Example RSS</title>
<item><title>First post</title><guid>post-1</guid><link>/posts/1</link>
<pubDate>Fri, 31 Jul 2026 12:00:00 GMT</pubDate>
<description><![CDATA[<p>Hello <strong>RSS</strong></p>]]></description></item>
</channel></rss>"""
    title, items = blogs.parse_feed(rss, "https://example.com/feed.xml", "Fallback")
    assert title == "Example RSS"
    assert len(items) == 1
    assert items[0].title == "First post"
    assert items[0].url == "https://example.com/posts/1"
    assert items[0].published_at == "2026-07-31T12:00:00+00:00"
    assert "Hello RSS" in items[0].text

    atom = """<feed xmlns="http://www.w3.org/2005/Atom">
<title>Example Atom</title><entry><id>tag:example,2026:2</id><title>Atom post</title>
<updated>2026-07-31T14:15:00Z</updated><link rel="alternate" href="/atom/2"/>
<summary type="html">Atom summary</summary></entry></feed>"""
    atom_title, atom_items = blogs.parse_feed(
        atom, "https://example.com/atom.xml", "Fallback"
    )
    assert atom_title == "Example Atom"
    assert atom_items[0].url == "https://example.com/atom/2"
    assert atom_items[0].published_at == "2026-07-31T14:15:00+00:00"

    with pytest.raises(ValueError, match="Invalid"):
        blogs.parse_feed("<rss>", "https://example.com/feed", "Fallback")
    with pytest.raises(ValueError, match="not an RSS"):
        blogs.parse_feed("<html/>", "https://example.com/feed", "Fallback")

    html = """<html><head>
<link rel="stylesheet" href="/style.css">
<link rel="alternate" type="application/rss+xml" href="/feed.xml">
<link rel="alternate" type="application/rss+xml" href="/feed.xml">
<link rel="alternate" type="application/atom+xml" href="http://127.0.0.1/feed">
</head><body><main>Snapshot fallback</main></body></html>"""
    assert blogs.discover_feed_urls(html, "https://example.com/blog") == (
        "https://example.com/feed.xml",
    )

    requests = []

    def discovered_handler(request):
        requests.append(str(request.url))
        if request.url.path == "/blog":
            return httpx.Response(200, headers={"content-type": "text/html"}, text=html)
        if request.url.path == "/feed.xml":
            return httpx.Response(
                200,
                headers={"content-type": "application/rss+xml", "etag": "feed-v1"},
                text=rss,
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<title>First post</title><main>Full website article</main>",
        )

    with httpx.Client(transport=httpx.MockTransport(discovered_handler)) as client:
        fetched = blogs.fetch_blog(
            client,
            {"url": "https://example.com/blog", "title": "Blog"},
        )
    assert fetched.method == "rss"
    assert fetched.feed_url == "https://example.com/feed.xml"
    assert fetched.etag == "feed-v1"
    assert requests == [
        "https://example.com/blog",
        "https://example.com/feed.xml",
        "https://example.com/posts/1",
    ]
    assert fetched.discovered_articles[0].url == "https://example.com/posts/1"
    assert "Full website article" in fetched.article_items[0].text

    def discovery_failure_handler(request):
        if request.url.path == "/blog":
            return httpx.Response(200, headers={"content-type": "text/html"}, text=html)
        return httpx.Response(500)

    with httpx.Client(transport=httpx.MockTransport(discovery_failure_handler)) as client:
        discovery_fallback = blogs.fetch_blog(
            client,
            {"url": "https://example.com/blog", "title": "Blog"},
        )
    assert discovery_fallback.method == "snapshot"
    assert discovery_fallback.feed_url == "https://example.com/feed.xml"

    feed_headers = {}

    def not_modified_handler(request):
        feed_headers.update(request.headers)
        return httpx.Response(304)

    with httpx.Client(transport=httpx.MockTransport(not_modified_handler)) as client:
        unchanged = blogs.fetch_blog(
            client,
            {
                "url": "https://example.com/blog",
                "title": "Blog",
                "feed_url": "https://example.com/feed.xml",
                "feed_etag": "feed-v1",
                "feed_last_modified": "today",
            },
        )
    assert unchanged.method == "rss" and unchanged.not_modified
    assert feed_headers["if-none-match"] == "feed-v1"
    assert feed_headers["if-modified-since"] == "today"

    def fallback_handler(request):
        if request.url.path == "/feed.xml":
            return httpx.Response(500)
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<title>Fallback page</title><main>Snapshot content</main>",
        )

    with httpx.Client(transport=httpx.MockTransport(fallback_handler)) as client:
        fallback = blogs.fetch_blog(
            client,
            {
                "url": "https://example.com/blog",
                "title": "Blog",
                "feed_url": "https://example.com/feed.xml",
            },
        )
    assert fallback.method == "snapshot"
    assert fallback.title == "Fallback page"
    assert fallback.snapshot_text == "Snapshot content"


def test_direct_feed_and_unusual_feed_values(monkeypatch):
    rss = """<rss><channel><item><description></description></item></channel></rss>"""
    title, items = blogs.parse_feed(rss, "https://example.com/feed", "Fallback")
    assert title == "Fallback"
    assert items[0].title == "Untitled post"
    assert items[0].url is None
    assert items[0].published_at is None

    odd_date = """<rss><channel><item><title>Odd</title><pubDate>not-a-date</pubDate>
<description>Body</description></item></channel></rss>"""
    _, odd_items = blogs.parse_feed(odd_date, "https://example.com/feed", "Fallback")
    assert odd_items[0].published_at == "not-a-date"

    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "application/xml"},
                text=odd_date,
            )
        )
    ) as client:
        direct = blogs.fetch_blog(client, {"url": "https://example.com/feed"})
    assert direct.method == "rss"
    assert direct.feed_url == "https://example.com/feed"

    monkeypatch.setattr(blogs, "MAX_TEXT_CHARS", 5)
    assert blogs._clean_lines("abcdefgh") == "abcde"


def test_blog_import_matches_creates_and_reports(tmp_path):
    csv_text = """Researcher / author,Current lab / organization,Primary applied-ML source,Primary source URL,Relationship to author,Secondary nearby source,Secondary source URL,Secondary relationship
Ada Lovelace,Institute,Ada Blog,https://example.com/blog,Direct,Lab,https://example.com/lab,Lab
Missing Person,Elsewhere,Missing,https://missing.example/blog,Direct,,,
Bad URL,Elsewhere,Bad,ftp://example.com,Direct,,,
"""
    db_path = tmp_path / "db.sqlite"
    with Database(db_path) as db:
        db.add_author(AuthorCandidate(openalex_id="A1", name="Ada Lovelace"))
        stats = import_blog_links(db, io.StringIO(csv_text), create_missing=True)
        assert stats.rows == 3
        assert stats.links == 3
        assert stats.authors_created == 2
        assert stats.invalid_urls == 1
        assert len(db.author_blogs("A1")) == 2
        created = db.find_authors_by_name("Missing Person")[0]
        assert str(created["openalex_id"]).startswith("LOCAL-")

        db.add_manual_author("Ada Lovelace")
        ambiguous = import_blog_links(db, io.StringIO(csv_text), create_missing=False)
        assert ambiguous.authors_ambiguous == 1

        missing_headers = io.StringIO("name,url\nAda,x\n")
        with pytest.raises(ValueError, match="CSV must contain"):
            import_blog_links(db, missing_headers)

        with pytest.raises(ValueError, match="CSV must contain"):
            import_blog_links(db, io.StringIO(""))

        padded_headers = io.StringIO(
            " Rank, Researcher / author, Primary source URL \n"
            " 1, Padded Person, https://example.com/padded \n"
        )
        padded = import_blog_links(db, padded_headers, create_missing=True)
        assert padded.rows == 1
        assert padded.links == 1
        assert padded.authors_created == 1

        path = tmp_path / "links.csv"
        path.write_text(csv_text, encoding="utf-8-sig")
        path_stats = import_blog_links_path(db, path, create_missing=False)
        assert path_stats.rows == 3


def test_blog_database_lifecycle_and_run_stats(app_config, author_candidate):
    responses = [
        httpx.Response(200, headers={"content-type": "text/html", "etag": "1"}, text="<main>First version</main>"),
        httpx.Response(200, headers={"content-type": "text/html", "etag": "2"}, text="<main>First version</main>"),
        httpx.Response(200, headers={"content-type": "text/html", "etag": "3"}, text="<main>Second version</main>"),
        httpx.Response(304),
    ]
    calls = []

    def handler(request):
        calls.append(request)
        response = responses.pop(0)
        response.request = request
        return response

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with Database(app_config.paths.database) as db:
        db.add_author(author_candidate)
        first_blog = db.add_author_blog("A123", "https://example.com/blog", label="Ada blog", relationship="Direct")
        assert db.add_author_blog("A123", "https://example.com/blog", label="Updated") == first_blog
        app = PaperWatch(app_config, db=db, openalex=NoPapers(), arxiv=None, llm=NoLLM(), pdf_client=client)
        first = app.run(today=date(2026, 7, 31))
        second = app.run(today=date(2026, 7, 31))
        third = app.run(today=date(2026, 7, 31))
        fourth = app.run(today=date(2026, 7, 31))
        assert first.blogs_checked == 1 and first.blog_updates_found == 0
        assert second.blog_updates_found == 0
        assert third.blog_updates_found == 1
        assert fourth.blog_updates_found == 0
        entries = db.list_blog_entries()
        assert len(entries) == 1 and entries[0]["change_kind"] == "changed"
        detail = db.get_blog_entry(int(entries[0]["id"]))
        assert detail and detail["author_names"] == "Ada Lovelace"
        assert db.get_blog_entry(9999) is None
        run = db.get_run(int(third.run_id))
        assert run and run["blogs_checked"] == 1 and run["blog_updates_found"] == 1
        assert db.remove_author_blog("A404", first_blog) is False
        assert db.remove_author_blog("A123", first_blog) is True
        assert db.author_blogs("A123") == []
    client.close()


def test_editing_and_marking_shared_blog_sources(tmp_path, author_candidate):
    second_author = AuthorCandidate(openalex_id="A456", name="Grace Hopper")
    with Database(tmp_path / "editable-blogs.sqlite") as db:
        db.add_author(author_candidate)
        db.add_author(second_author)
        blog_id = db.add_author_blog(
            "A123",
            "https://example.com/original",
            label="Shared source",
        )
        assert db.add_author_blog("A456", "https://example.com/original") == blog_id
        entry_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Historical post",
            content_sha256="old",
            normalized_text="old text",
            diff_text="",
            change_kind="baseline",
            previous_entry_id=None,
        )
        db._conn.execute(
            """
            UPDATE blogs SET enabled=0, etag='etag', last_modified='date',
                feed_url='https://example.com/feed', feed_etag='feed-etag',
                feed_last_modified='feed-date', feed_initialized=1,
                feed_discovery_checked=1, last_fetch_method='rss',
                last_checked_at='2026-08-01', last_status=500, last_error='failed'
            WHERE id=?
            """,
            (blog_id,),
        )
        db._conn.commit()

        first = db.author_blogs("A123")[0]
        assert first["linked_author_count"] == 2
        assert db.update_author_blog_url(
            "A123", blog_id, "https://example.com/replacement"
        ) is True
        for author_id in ("A123", "A456"):
            row = db.author_blogs(author_id)[0]
            assert row["id"] == blog_id
            assert row["url"] == "https://example.com/replacement"
            assert row["enabled"] == 1
            assert row["feed_url"] is None
            assert row["last_checked_at"] is None
            assert row["last_error"] is None
        assert db.get_blog_entry(entry_id) is not None

        assert db.set_author_blog_broken("A456", blog_id, True) is True
        broken = db.author_blogs("A123")[0]
        assert broken["enabled"] == 0
        assert broken["last_error"] == "Marked broken by user"
        assert db.list_enabled_blogs() == []
        assert db.set_author_blog_broken("A123", blog_id, False) is True
        assert db.author_blogs("A456")[0]["enabled"] == 1

        assert db.update_author_blog_url(
            "A123", blog_id, "https://example.com/replacement"
        ) is True
        duplicate_id = db.add_author_blog("A123", "https://example.com/duplicate")
        with pytest.raises(ValueError, match="already tracked"):
            db.update_author_blog_url(
                "A123", blog_id, "https://example.com/duplicate"
            )
        assert duplicate_id != blog_id
        assert db.update_author_blog_url(
            "A404", blog_id, "https://example.com/missing"
        ) is False
        assert db.set_author_blog_broken("A404", blog_id, True) is False
        with pytest.raises(ValueError, match="required"):
            db.update_author_blog_url("A123", blog_id, " ")


def test_rss_items_are_imported_without_triggering_initial_updates(app_config, author_candidate):
    page = """<html><head><link rel="alternate" type="application/rss+xml" href="/feed.xml"></head><body><main>Page</main></body></html>"""

    def rss(*items):
        body = "".join(
            f"<item><guid>{identifier}</guid><title>{title}</title>"
            f"<link>https://example.com/{identifier}</link>"
            f"<description>{text}</description></item>"
            for identifier, title, text in items
        )
        return f"<rss><channel><title>Author feed</title>{body}</channel></rss>"

    feed_versions = [
        rss(("one", "First", "First body"), ("two", "Second", "Second body")),
        rss(
            ("one", "First", "First body"),
            ("two", "Second", "Second body"),
            ("three", "Third", "Third body"),
        ),
        rss(
            ("one", "First", "First body"),
            ("two", "Second revised", "Updated body"),
            ("three", "Third", "Third body"),
        ),
    ]
    page_fetched = False

    article_titles = {"/one": "First", "/two": "Second", "/three": "Third"}

    def handler(request):
        nonlocal page_fetched
        if request.url.path == "/blog":
            page_fetched = True
            return httpx.Response(200, headers={"content-type": "text/html"}, text=page)
        if request.url.path == "/feed.xml":
            return httpx.Response(
                200,
                headers={"content-type": "application/rss+xml"},
                text=feed_versions.pop(0),
            )
        title = article_titles[request.url.path]
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text=f"<title>{title}</title><main>{title} website body</main>",
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with Database(app_config.paths.database) as db:
        db.add_author(author_candidate)
        db.add_author_blog("A123", "https://example.com/blog")
        app = PaperWatch(
            app_config,
            db=db,
            openalex=NoPapers(),
            arxiv=None,
            llm=NoLLM(),
            pdf_client=client,
        )
        first = app.run(today=date(2026, 7, 31))
        assert page_fetched
        assert first.blog_updates_found == 0
        blog = db.author_blogs("A123")[0]
        assert blog["feed_url"] == "https://example.com/feed.xml"
        assert blog["feed_initialized"] == 1
        assert blog["last_fetch_method"] == "rss"
        initial_entries = db.list_blog_entries()
        assert len(initial_entries) == 2
        assert {entry["title"] for entry in initial_entries} == {"First", "Second"}
        assert {entry["entry_kind"] for entry in initial_entries} == {"rss"}
        assert {entry["change_kind"] for entry in initial_entries} == {"changed"}
        assert any("website body" in entry["normalized_text"] for entry in initial_entries)

        second = app.run(today=date(2026, 7, 31))
        assert second.blog_updates_found == 1
        new_entries = db.list_blog_entries()
        assert len(new_entries) == 3
        third_entries = [entry for entry in new_entries if entry["title"] == "Third"]
        assert {entry["entry_kind"] for entry in third_entries} == {"rss"}
        assert {entry["source_url"] for entry in third_entries} == {
            "https://example.com/three"
        }

        third = app.run(today=date(2026, 7, 31))
        assert third.blog_updates_found == 1
        changed = db.list_blog_entries()[0]
        assert changed["title"] == "Second revised"
        assert changed["entry_kind"] == "rss"
        assert changed["previous_entry_id"] is not None
        assert "Updated body" in changed["diff_text"]
    client.close()


def test_blog_fetch_error_is_recorded(app_config, author_candidate):
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500)))
    with Database(app_config.paths.database) as db:
        db.add_author(author_candidate)
        blog_id = db.add_author_blog("A123", "https://example.com/blog")
        app = PaperWatch(app_config, db=db, openalex=NoPapers(), arxiv=None, llm=NoLLM(), pdf_client=client)
        stats = app.run(today=date(2026, 7, 31))
        assert stats.status == "partial"
        assert stats.errors_count == 1
        blog = db.author_blogs("A123")[0]
        assert blog["id"] == blog_id and blog["last_error"]
        assert db.recent_errors(1)[0]["stage"] == "blog_fetch"
    client.close()


def test_manual_author_and_import_missing_branches(tmp_path):
    with Database(tmp_path / "branches.db") as db:
        with pytest.raises(ValueError, match="required"):
            db.add_manual_author("   ")
        with pytest.raises(ValueError, match="too long"):
            db.add_manual_author("x" * 301)
        assert db.find_authors_by_name("   ") == []
        db.add_author(AuthorCandidate(openalex_id="A1", name="Ada"))
        with pytest.raises(ValueError, match="Blog URL"):
            db.add_author_blog("A1", "")
        with pytest.raises(ValueError, match="Author not found"):
            db.add_author_blog("A404", "https://example.com")
        stats = import_blog_links(
            db,
            io.StringIO(
                "Researcher / author,Primary source URL\n"
                ",https://example.com/blank\n"
                "Missing,https://example.com/missing\n"
            ),
            create_missing=False,
        )
        assert stats.authors_missing == 2


def test_schema_10_blog_tables_migrate_for_rss(tmp_path):
    path = tmp_path / "v10.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE blogs (
            id INTEGER PRIMARY KEY, url TEXT NOT NULL UNIQUE, title TEXT,
            enabled INTEGER NOT NULL DEFAULT 1, etag TEXT, last_modified TEXT,
            last_checked_at TEXT, last_status INTEGER, last_error TEXT,
            added_at TEXT NOT NULL
        );
        CREATE TABLE blog_entries (
            id INTEGER PRIMARY KEY, blog_id INTEGER NOT NULL,
            previous_entry_id INTEGER, title TEXT NOT NULL,
            content_sha256 TEXT NOT NULL, normalized_text TEXT NOT NULL,
            diff_text TEXT NOT NULL DEFAULT '', change_kind TEXT NOT NULL,
            fetched_at TEXT NOT NULL
        );
        PRAGMA user_version = 10;
        """
    )
    conn.close()

    with Database(path) as db:
        blog_columns = {
            row["name"] for row in db._conn.execute("PRAGMA table_info(blogs)")
        }
        entry_columns = {
            row["name"] for row in db._conn.execute("PRAGMA table_info(blog_entries)")
        }
        assert {"feed_url", "feed_initialized", "last_fetch_method"} <= blog_columns
        assert {"entry_kind", "external_id", "source_url", "published_at"} <= entry_columns


def test_html_date_handles_malformed_metadata_nested_json_and_visible_fallback():
    assert blogs._html_first_date(
        """<html><head>
        <meta name="description" content="2020-01-01">
        <meta name="date" content="not-a-date">
        <script type="application/ld+json">not json</script>
        <script type="application/ld+json">[{"nested":{"dateCreated":"2023-02-04"}}]</script>
        </head><body><time>n/a</time></body></html>"""
    ) == "2023-02-04"
    assert blogs._html_first_date("<p>Published September 14, 2022</p>") == "2022-09-14"
    assert blogs._html_first_date("<p>No publication date here</p>") is None


def test_link_label_inline_markdown_and_clean_lines_edge_cases():
    class Broken:
        pass

    assert blogs._link_label(Broken()) == ""
    soup = blogs.BeautifulSoup(
        '<div><a aria-label="Fallback label"></a><a title="avatar icon"></a></div>',
        "html.parser",
    )
    anchors = soup.find_all("a")
    assert blogs._link_label(anchors[0]) == "Fallback label"
    assert blogs._link_label(anchors[1]) == ""

    tag = blogs.BeautifulSoup(
        '<p>Hello <a href="https://example.com"><img src="https://example.com/a.png" alt=" Plot "></a>'
        '<img src="https://example.com/b.png" alt=" Figure "> '
        '<a href="https://example.com/read"> Read more </a><br>next , line</p>',
        "html.parser",
    ).p
    rendered = blogs._inline_markdown(tag)
    assert "![Plot](https://example.com/a.png)" in rendered
    assert "![Figure](https://example.com/b.png)" in rendered
    assert "[Read more](https://example.com/read)" in rendered
    assert "\nnext, line" in rendered

    assert blogs._clean_lines(" one  two \n one two \n three ") == "one two\nthree"
    with pytest.raises(ValueError, match="readable text"):
        blogs._clean_lines(" \n \t")


def test_article_candidate_filtering_pattern_external_and_deduplication():
    root = blogs.BeautifulSoup(
        """<main>
        <article><a href="/posts/2026/good">Good post</a><p>Published 2026-04-03. Useful summary.</p></article>
        <article><a href="/posts/2026/good">Duplicate post</a></article>
        <a href="/about">About us</a>
        <a href="https://other.example/posts/remote">Remote post</a>
        <a href="https://doi.org/10.1000/example">Paper DOI</a>
        <a href="/plain">Plain link</a>
        </main>""",
        "html.parser",
    )
    normal = blogs._article_candidates(root, base_url="https://example.com/blog")
    assert [item.url for item in normal] == ["https://example.com/posts/2026/good"]

    external = blogs._article_candidates(
        root,
        base_url="https://example.com/blog",
        allow_external=True,
        pattern=r"(?:remote|doi\.org)",
    )
    assert {item.url for item in external} == {
        "https://other.example/posts/remote",
        "https://doi.org/10.1000/example",
    }

    assert blogs._should_crawl_article_index(
        configured_mode="article_index", page_url="https://example.com/post/x", pattern=None, candidates=()
    )
    assert not blogs._should_crawl_article_index(
        configured_mode="auto", page_url="https://example.com/post/x", pattern=None, candidates=()
    )
    candidate = normal[:1]
    assert blogs._should_crawl_article_index(
        configured_mode="feed", page_url="https://example.com/post/x", pattern=None, candidates=candidate
    )
    assert not blogs._should_crawl_article_index(
        configured_mode="article", page_url="https://example.com/blog", pattern=None, candidates=candidate
    )
    assert blogs._should_crawl_article_index(
        configured_mode="auto", page_url="https://example.com/post/x", pattern=r"post", candidates=candidate
    )
    assert blogs._should_crawl_article_index(
        configured_mode="auto", page_url="https://example.com/blog", pattern=None, candidates=candidate
    )


def test_snapshot_and_meaningful_link_change_modes_cover_semantic_edges():
    content, links = blogs._split_snapshot(
        "URL: https://example.com/direct\nBody\nOutbound links:\n"
        "- Docs -> https://example.com/docs\n"
        "bad line\n- Empty -> \n- Docs -> https://example.com/docs/v2"
    )
    assert content == ["URL: https://example.com/direct", "Body"]
    assert links == {
        "docs": frozenset({"https://example.com/docs", "https://example.com/docs/v2"}),
        "__direct__": frozenset({"https://example.com/direct"}),
    }
    assert blogs._split_snapshot("Body only") == (["Body only"], None)

    old = {"docs": frozenset({"a"})}
    same_urls_different_labels = {"other": frozenset({"a"})}
    assert not blogs._meaningful_link_change(old, same_urls_different_labels, source_mode="auto", content_changed=False)
    assert blogs._meaningful_link_change(old, {"docs": frozenset({"b"})}, source_mode="article", content_changed=False)
    assert not blogs._meaningful_link_change(old, {"new": frozenset({"b"})}, source_mode="docs", content_changed=False)
    assert blogs._meaningful_link_change(
        old,
        {"new": frozenset({"b", "c", "d"})},
        source_mode="docs",
        content_changed=False,
    )
    assert not blogs._meaningful_link_change(old, {"new": frozenset({"b"})}, source_mode="publications", content_changed=False)
    assert blogs._meaningful_link_change(old, {"new": frozenset({"b"})}, source_mode="publications", content_changed=True)
    assert not blogs._meaningful_link_change({}, {"new": frozenset({"b"})}, source_mode="auto", content_changed=False)
    assert blogs._meaningful_link_change(old, {"new": frozenset({"b", "c"})}, source_mode="auto", content_changed=False)
    assert not blogs._meaningful_link_change(None, {"x": frozenset({"a"})}, source_mode="auto", content_changed=True)


def test_same_title_new_path_reuses_entry_and_preserves_user_history(
    app_config, author_candidate
):
    old_text = "Title: Stable post\nURL: https://example.com/posts/old\nOld body"
    new_text = "Title: Stable post\nURL: https://example.com/writing/new-path\nNew body"
    with Database(app_config.paths.database) as db:
        db.add_author(author_candidate)
        blog_id = db.add_author_blog("A123", "https://example.com/blog")
        entry_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Stable post",
            content_sha256=hashlib.sha256(old_text.encode()).hexdigest(),
            normalized_text=old_text,
            diff_text="",
            change_kind="changed",
            previous_entry_id=None,
            entry_kind="rss",
            external_id="old-guid",
            source_url="https://example.com/posts/old",
        )
        db.set_object_notes("blog", entry_id, "Keep this note")
        query_id = db.create_saved_query(name="Existing summary", prompt="Summarize")
        run_id = db.create_query_runs(
            batch_id="existing-summary",
            query_id=query_id,
            profiles=[("default", "none", "none")],
            context_window_tokens=1000,
            selection={"blog_entry_ids": [entry_id]},
        )[0]
        db.start_query_run(run_id)
        db.finish_query_run(
            run_id,
            matched_count=1,
            input_chars=len(old_text),
            output={
                "summary": "Old summary",
                "items": [{
                    "blog_entry_id": entry_id,
                    "content_type": "blog",
                    "title": "Stable post",
                    "output": "Old summary",
                }],
            },
        )

        app = PaperWatch(
            app_config, db=db, openalex=NoPapers(), arxiv=None, llm=NoLLM()
        )
        row = app._prepare_blog_fetch_rows(db.list_enabled_blogs())[0]
        fetched = blogs.BlogFetch(
            method="rss",
            title="Feed",
            status_code=200,
            feed_items=(
                blogs.FeedItem(
                    external_id="new-guid",
                    title="Stable post",
                    url="https://example.com/writing/new-path",
                    published_at="2026-08-08",
                    text=new_text,
                    sha256=hashlib.sha256(new_text.encode()).hexdigest(),
                ),
            ),
            source_mode="feed",
        )
        from paperwatch.pipeline import RunStats
        run_stats = RunStats(run_id=db.start_run())
        app._apply_blog_fetch(run_stats, row, fetched, None)

        entries = db.list_blog_entries()
        assert len(entries) == 1
        current = entries[0]
        assert int(current["id"]) == entry_id
        assert current["source_url"] == "https://example.com/writing/new-path"
        assert current["external_id"] == "new-guid"
        assert current["normalized_text"] == new_text
        assert current["notes"] == "Keep this note"
        assert db.query_runs_for_blog_entry(entry_id)[0]["id"] == run_id
        assert run_stats.new_blog_entry_ids == [entry_id]
        app.close()


def test_same_title_requires_same_author_and_site(tmp_path, author_candidate):
    other = AuthorCandidate(openalex_id="A456", name="Other Author")
    with Database(tmp_path / "blog-title-identity.sqlite") as db:
        db.add_author(author_candidate)
        db.add_author(other)
        first_blog = db.add_author_blog("A123", "https://example.com/blog")
        second_blog = db.add_author_blog("A456", "https://example.com/other")
        db.save_blog_entry(
            blog_id=first_blog,
            title="Shared title",
            content_sha256="one",
            normalized_text="one",
            diff_text="",
            change_kind="changed",
            previous_entry_id=None,
            source_url="https://example.com/old/path",
        )
        assert db.matching_blog_entry_by_title(
            second_blog,
            title="Shared title",
            source_url="https://example.com/new/path",
        ) is None
        db.add_author_blog("A123", "https://example.com/other")
        matched = db.matching_blog_entry_by_title(
            second_blog,
            title="Shared title",
            source_url="https://example.com/new/path",
        )
        assert matched is not None and int(matched["blog_id"]) == first_blog
        db.save_blog_entry(
            blog_id=second_blog,
            title="Shared title",
            content_sha256="two",
            normalized_text="two",
            diff_text="",
            change_kind="changed",
            previous_entry_id=None,
            source_url="https://example.com/new/path",
        )
        assert db.count_blog_entries(dedupe_titles=True) == 1


def test_schema_31_migration_collapses_existing_same_title_rows(tmp_path, author_candidate):
    path = tmp_path / "title-dedupe-migration.sqlite"
    with Database(path) as db:
        db.add_author(author_candidate)
        blog_id = db.add_author_blog("A123", "https://example.com/blog")
        old_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Same title",
            content_sha256="old",
            normalized_text="Old content",
            diff_text="",
            change_kind="changed",
            previous_entry_id=None,
            entry_kind="rss",
            external_id="old-guid",
            source_url="https://example.com/old",
            published_at="2026-01-01",
        )
        db.set_object_notes("blog", old_id, "Persistent note")
        new_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Same title",
            content_sha256="new",
            normalized_text="Newest content",
            diff_text="+Newest content",
            change_kind="changed",
            previous_entry_id=old_id,
            entry_kind="rss",
            external_id="new-guid",
            source_url="https://example.com/new-path",
            published_at="2026-08-08",
        )
        db.add_blog_entry_tag(new_id, "migrated-tag")
        db._conn.execute("PRAGMA user_version = 30")
        db._conn.commit()

    with Database(path) as db:
        visible = db.list_blog_entries()
        assert len(visible) == 1
        assert int(visible[0]["id"]) == old_id
        assert visible[0]["normalized_text"] == "Newest content"
        assert visible[0]["source_url"] == "https://example.com/new-path"
        assert visible[0]["notes"] == "Persistent note"
        assert "migrated-tag" in {row["name"] for row in db.blog_entry_tags(old_id)}
        hidden = db._conn.execute(
            "SELECT change_kind FROM blog_entries WHERE id=?", (new_id,)
        ).fetchone()
        assert hidden["change_kind"] == "baseline"
        assert db._conn.execute("PRAGMA user_version").fetchone()[0] == 32
