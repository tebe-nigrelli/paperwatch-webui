from __future__ import annotations

import hashlib
import json
import threading
from datetime import date

import httpx
import pytest
from bs4 import BeautifulSoup

from paperwatch import blogs
from paperwatch import db as db_module
from paperwatch.blogs import ArticleItem, ArticleLink, BlogFetch
from paperwatch.db import Database, SCHEMA_VERSION
from paperwatch.models import AuthorCandidate
from paperwatch.pipeline import PaperWatch, RunStats


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


def test_source_mode_validation_inference_and_link_helpers(monkeypatch):
    assert blogs.normalize_source_mode(None) == "auto"
    assert blogs.normalize_source_mode("article-index") == "article_index"
    with pytest.raises(ValueError, match="Unknown"):
        blogs.normalize_source_mode("magic")

    assert blogs.validate_content_selector(None) is None
    assert blogs.validate_content_selector(" main > article ") == "main > article"
    with pytest.raises(ValueError, match="too long"):
        blogs.validate_content_selector("a" * 501)
    with pytest.raises(ValueError, match="invalid"):
        blogs.validate_content_selector("main[")
    assert blogs.validate_article_url_pattern("") is None
    assert blogs.validate_article_url_pattern(r"/posts/\d+") == r"/posts/\d+"
    with pytest.raises(ValueError, match="too long"):
        blogs.validate_article_url_pattern("x" * 501)
    with pytest.raises(ValueError, match="invalid"):
        blogs.validate_article_url_pattern("[")

    assert blogs.infer_source_mode("https://github.com/org/repo") == "github"
    assert blogs.infer_source_mode("https://example.com", "GitHub releases") == "github"
    assert blogs.infer_source_mode("https://youtube.com/@one") == "video"
    assert blogs.infer_source_mode("https://example.com", "Video channel") == "video"
    assert blogs.infer_source_mode("https://example.com", "Living technical notes") == "docs"
    assert blogs.infer_source_mode("https://example.com", "Research publication feed") == "publications"
    assert blogs.infer_source_mode("https://example.com", "Email newsletter") == "feed"
    assert blogs.infer_source_mode("https://example.com", "Personal blog") == "auto"
    assert blogs._row_value({}, "missing", "fallback") == "fallback"

    assert blogs._normalize_link_url(
        "/post?utm_source=x&keep=1&fbclid=y#part", "https://EXAMPLE.com:443/blog"
    ) == "https://example.com/post?keep=1"
    assert blogs._normalize_link_url("https://example.com:8443/a", "https://example.com") == "https://example.com:8443/a"
    assert blogs._normalize_link_url("mailto:a@example.com", "https://example.com") is None
    assert blogs._normalize_link_url("http://user:pass@example.com/a", "https://example.com") is None
    assert blogs._normalize_link_url("http://127.0.0.1/a", "https://example.com") is None
    assert blogs._normalize_link_url("http://localhost/a", "https://example.com") is None
    assert blogs._normalize_link_url("http://example.com:bad/a", "https://example.com") is None

    soup = BeautifulSoup(
        """<main>
        <a href='/post?utm_medium=x'>Useful post</a>
        <a href='/post'>Useful post</a>
        <a href='/other' aria-label='ARIA label'></a>
        <a href='https://twitter.com/example'>Twitter</a>
        <a href='/asset.pdf'>Paper PDF</a>
        <a href='/'>Home page</a>
        <a href='/share/item'>Share item</a>
        <a href='/avatar'>Person's profile picture</a>
        </main>""",
        "html.parser",
    )
    root = soup.main
    assert root is not None
    assert blogs._link_label(object()) == ""
    assert blogs._link_label(root.find_all("a")[2]) == "ARIA label"
    assert blogs._link_label(BeautifulSoup(f"<a>{'x' * 301}</a>", "html.parser").a) == ""
    candidate_soup = BeautifulSoup(
        """<main><ul><li><a href='/products/widget'>Widget</a></li></ul>
        <article><a href='/blog/real-post'>Real post</a></article></main>""",
        "html.parser",
    )
    candidates = blogs._article_candidates(
        candidate_soup, base_url="https://example.com/blog/"
    )
    assert [item.url for item in candidates] == ["https://example.com/blog/real-post"]

    pairs = blogs._outbound_link_pairs(root, "https://example.com/blog")
    assert pairs == (
        ("Useful post", "https://example.com/post"),
        ("ARIA label", "https://example.com/other"),
    )
    assert blogs._outbound_link_pairs(object(), "https://example.com") == ()
    assert blogs._ignored_link("Login", "https://example.com/login", "https://example.com")
    assert blogs._ignored_link("Social", "https://www.facebook.com/x", "https://example.com")
    assert blogs._ignored_link("Image", "https://example.com/x.png", "https://example.com")
    assert blogs._ignored_link("Home page", "https://example.com/", "https://example.com")
    assert not blogs._ignored_link("Article", "https://example.com/posts/one", "https://example.com")

    with_links = blogs._text_with_links("Body", pairs)
    assert blogs.OUTBOUND_LINKS_HEADER in with_links
    assert blogs._text_with_links("Body", ()) == "Body"
    assert blogs._content_sha256("Body", ("https://example.com/a",)) != blogs._content_sha256("Body")
    assert blogs.decode_outbound_links('["a", "a", "b"]') == ("a", "b")
    assert blogs.decode_outbound_links(None) is None
    assert blogs.decode_outbound_links("bad") is None
    assert blogs.decode_outbound_links("{}") is None

    title, text, links = blogs._extract_blog_document(
        "<title>T</title><body><div class='content'>Text <a href='/post'>Post</a></div></body>",
        "Fallback",
        "https://example.com/blog",
        ".content",
    )
    assert title == "T" and "Text" in text and links == ("https://example.com/post",)
    with pytest.raises(ValueError, match="did not match"):
        blogs._extract_blog_document("<main>Text</main>", "F", "https://example.com", ".missing")
    assert blogs._document_root(BeautifulSoup("<article>A</article>", "html.parser"), None).name == "article"

    assert blogs._html_text("<p>Hello <b>world</b></p>")[0] == "Hello world"
    assert blogs._published_at("") is None
    assert blogs._published_at("2026-08-01T12:00:00") == "2026-08-01T12:00:00+00:00"
    assert blogs._published_at("not-a-real-date") == "not-a-real-date"
    assert blogs.extract_first_date("Jul 18, 2022") == "2022-07-18"
    assert blogs.extract_first_date("June, 2020") == "2020-06"
    assert blogs.extract_first_date("9/10/2025") == "2025-09-10"
    assert blogs.extract_first_date(" 2023-09-22 ") == "2023-09-22"
    assert blogs.extract_first_date("ertApr 14, 20265554Share") == "2026-04-14"
    assert blogs.extract_first_date("31 Dec 2024") == "2024-12-31"
    assert blogs.extract_first_date("2/30/2025 then Jul 1, 2020") == "2020-07-01"
    assert blogs.extract_first_date("not a date") is None

    monkeypatch.setattr(blogs, "MAX_TEXT_CHARS", 20)
    assert len(blogs._text_with_links("Body", pairs)) == 20


def test_feed_identity_collisions_and_embedded_links():
    rss = """<rss><channel><title>Collision feed</title>
    <item><title>First</title><guid>same</guid><link>https://hamel.dev/</link><pubDate>2026-07-01</pubDate><description><![CDATA[<p>One <a href='https://outside.test/a?utm_source=x'>source</a></p>]]></description></item>
    <item><title>Second</title><guid>same</guid><link>https://hamel.dev/</link><pubDate>2026-07-02</pubDate><description>Two</description></item>
    <item><title>Duplicate fallback</title><description><![CDATA[<script>x</script>]]></description></item>
    <item><title>Duplicate fallback</title><description><![CDATA[<script>x</script>]]></description></item>
    <item><title>GUID only</title><guid>tag:example,2026:five</guid><description>Five</description></item>
    </channel></rss>"""
    title, items = blogs.parse_feed(rss, "https://hamel.dev/index.xml", "Fallback")
    assert title == "Collision feed"
    assert len(items) == 5
    assert items[0].url is None and items[1].url is None
    assert items[0].external_id != items[1].external_id
    assert items[2].external_id != items[3].external_id
    assert items[4].url is None
    assert items[0].outbound_links == ("https://outside.test/a",)
    assert "Outbound links:" in items[0].text

    unique = """<rss><channel><item><title>Unique</title><guid>https://example.com/guid</guid>
    <link>https://example.com/posts/one</link><description>Body</description></item></channel></rss>"""
    _, unique_items = blogs.parse_feed(unique, "https://example.com/feed.xml", "F")
    assert unique_items[0].url == "https://example.com/posts/one"

    atom = """<feed xmlns='http://www.w3.org/2005/Atom'><entry><title>Atom</title>
    <link rel='self' href='/self'/><link rel='alternate' href='/post'/><content><![CDATA[<a href='/ref'>Reference</a>]]></content>
    </entry></feed>"""
    _, atom_items = blogs.parse_feed(atom, "https://example.com/feed", "Atom feed")
    assert atom_items[0].url == "https://example.com/post"
    assert atom_items[0].outbound_links == ("https://example.com/ref",)


def test_feed_scope_conventional_urls_and_response_guards(monkeypatch):
    assert blogs._scope_tokens("https://example.com/authors/Alice/?tag=ML") == ("alice", "ml")
    assert blogs._scope_tokens("https://example.com/blog") == ()
    assert blogs._is_scoped_page("https://example.com/category/systems/")
    scoped = blogs.feed_scope_score(
        "https://example.com/author/alice/", "https://example.com/author/alice/feed/"
    )
    global_score = blogs.feed_scope_score(
        "https://example.com/author/alice/", "https://example.com/feed/"
    )
    assert scoped > global_score and scoped >= 25 and global_score < 25
    assert blogs.feed_scope_score("https://a.example/blog", "https://b.example/feed") < 0
    conventional = blogs._conventional_feed_urls("https://example.com/blog/")
    assert "https://example.com/blog/feed/" in conventional
    assert "https://example.com/feed.xml" in conventional
    assert len(conventional) == len(set(conventional))
    assert blogs._headers("etag", "date") == {
        "If-None-Match": "etag",
        "If-Modified-Since": "date",
    }
    assert blogs._headers(None, None) == {}
    request = httpx.Request("GET", "https://example.com")
    unchanged = httpx.Response(304, request=request)
    assert blogs._checked_response(unchanged) is unchanged
    monkeypatch.setattr(blogs, "MAX_RESPONSE_BYTES", 1)
    with pytest.raises(ValueError, match="exceeds"):
        blogs._checked_response(httpx.Response(200, content=b"xx", request=request))


def test_article_candidate_detection_and_patterns():
    html = """<main>
    <article class='post-card'><a href='/blog/one'>First article</a><p>Summary one</p></article>
    <article class='post-card'><a href='/blog/one'>First article duplicate</a></article>
    <article><a href='/special/two'>Second special article</a></article>
    <article><a href='https://doi.org/10.1/example'>External publication</a></article>
    <a href='/categories/ml'>ML category</a>
    <a href='https://twitter.com/x'>Social link</a>
    </main>"""
    soup = BeautifulSoup(html, "html.parser")
    root = soup.main
    assert root is not None
    normal = blogs._article_candidates(root, base_url="https://example.com/blog")
    assert [item.url for item in normal] == [
        "https://example.com/blog/one",
        "https://example.com/special/two",
    ]
    patterned = blogs._article_candidates(
        root,
        base_url="https://example.com/blog",
        pattern=r"/special/",
    )
    assert [item.url for item in patterned] == ["https://example.com/special/two"]
    publications = blogs._article_candidates(
        root,
        base_url="https://example.com/publications",
        allow_external=True,
    )
    assert "https://doi.org/10.1/example" in {item.url for item in publications}
    assert blogs._article_candidates(object(), base_url="https://example.com") == ()
    assert blogs._article_score(object(), url="https://other.test/x", base_url="https://example.com") == 0


def test_scoped_feed_selection_and_direct_rejection():
    page = """<html><head>
    <link rel='alternate' type='application/rss+xml' href='/feed/'>
    <link rel='alternate' type='application/rss+xml' href='/author/alice/feed/'>
    </head><main>Author posts</main></html>"""
    rss = "<rss><channel><title>Alice</title><item><title>Post</title><link>/posts/1</link><description>Body</description></item></channel></rss>"
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/author/alice/":
            return httpx.Response(200, headers={"content-type": "text/html"}, text=page)
        if request.url.path == "/author/alice/feed/":
            return httpx.Response(200, headers={"content-type": "application/rss+xml"}, text=rss)
        if request.url.path == "/posts/1":
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="<title>Post</title><main>Website body</main>",
            )
        raise AssertionError(f"global feed should not be requested: {request.url}")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        fetched = blogs.fetch_blog(
            client,
            {"url": "https://example.com/author/alice/", "source_mode": "auto"},
        )
    assert fetched.method == "rss"
    assert fetched.feed_url == "https://example.com/author/alice/feed/"
    assert any("Rejected out-of-scope" in item for item in fetched.diagnostics)
    assert seen == ["/author/alice/", "/author/alice/feed/", "/posts/1"]
    assert fetched.article_items and "Website body" in fetched.article_items[0].text

    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(304))) as client:
        with pytest.raises(ValueError, match="out-of-scope"):
            blogs._fetch_feed_url(
                client,
                feed_url="https://example.com/feed/",
                title="Alice",
                page_url="https://example.com/author/alice/",
            )


def test_article_index_publications_docs_and_selector_fetch(monkeypatch):
    monkeypatch.setattr(blogs, "MAX_ARTICLE_FETCHES", 2)
    listing = """<title>Index</title><main class='posts'>
    <article><a href='/blog/new'>New article</a><p>New summary</p></article>
    <article><a href='/blog/old'>Old article</a><p>Old summary</p></article>
    </main>"""
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path == "/blog":
            return httpx.Response(200, headers={"content-type": "text/html"}, text=listing)
        if request.url.path == "/blog/new":
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="<title>New full</title><article>Full body <a href='https://outside.test/ref'>Reference</a></article>",
            )
        if request.url.path == "/blog/off-index":
            return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"pdf")
        raise AssertionError(request.url)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        fetched = blogs.fetch_blog(
            client,
            {
                "url": "https://example.com/blog",
                "source_mode": "article_index",
                "content_selector": ".posts",
                "article_url_pattern": r"/blog/",
                "_known_article_urls": (
                    "https://example.com/blog/old",
                    "https://example.com/blog/off-index",
                ),
                "article_cursor": 1,
            },
        )
    assert fetched.source_mode == "article_index"
    assert {item.url for item in fetched.discovered_articles} == {
        "https://example.com/blog/new",
        "https://example.com/blog/old",
    }
    assert [item.url for item in fetched.article_items] == ["https://example.com/blog/new"]
    assert any("off-index" in item for item in fetched.diagnostics)
    assert fetched.article_cursor_next == 0
    assert requests == ["/blog", "/blog/new", "/blog/off-index"]

    publication_page = """<main><article><a href='https://doi.org/10.1/paper'>Paper title</a></article></main>"""
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, headers={"content-type": "text/html"}, text=publication_page)
        )
    ) as client:
        publications = blogs.fetch_blog(
            client,
            {"url": "https://example.com/publications", "source_mode": "publications"},
        )
    assert publications.source_mode == "publications"
    assert publications.discovered_articles[0].url == "https://doi.org/10.1/paper"
    assert [item.url for item in publications.article_items] == [
        "https://doi.org/10.1/paper"
    ]


def test_aggressive_auto_blog_discovery_crawls_non_index_page_and_more_articles():
    links = "".join(
        f"<article><a href='/writing/post-{index}'>Research note number {index}</a>"
        f"<p>Detailed research update {index}</p></article>"
        for index in range(12)
    )
    listing = f"<html><head><title>Alice</title></head><body><main>{links}</main></body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/alice":
            return httpx.Response(200, headers={"content-type": "text/html"}, text=listing)
        if request.url.path.startswith("/writing/post-"):
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text=f"<article><h1>{request.url.path}</h1><p>Complete article body.</p></article>",
            )
        return httpx.Response(404, headers={"content-type": "text/html"}, text="missing")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        fetched = blogs.fetch_blog(
            client,
            {
                "url": "https://example.com/alice",
                "source_mode": "auto",
                "_aggressive_discovery": True,
                "_feed_candidate_limit": 16,
                "_article_fetch_limit": 24,
            },
        )

    assert fetched.source_mode == "article_index"
    assert len(fetched.discovered_articles) == 12
    assert len(fetched.article_items) == 12

    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="<main><section class='docs'>Documentation text</section><aside>Noise</aside></main>",
            )
        )
    ) as client:
        docs = blogs.fetch_blog(
            client,
            {
                "url": "https://example.com/docs",
                "source_mode": "docs",
                "content_selector": ".docs",
            },
        )
    assert docs.source_mode == "docs" and docs.snapshot_text == "Documentation text"

    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, headers={"content-type": "text/html"}, text="<main>Text</main>")
        )
    ) as client:
        with pytest.raises(ValueError, match="did not match"):
            blogs.fetch_blog(
                client,
                {
                    "url": "https://example.com/docs",
                    "source_mode": "docs",
                    "content_selector": ".missing",
                },
            )


def test_feed_video_and_direct_xml_fallback_paths():
    rss = "<rss><channel><title>Feed</title><item><title>One</title><link>/one</link><description>Body</description></item></channel></rss>"
    calls: list[str] = []

    def feed_handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/blog":
            return httpx.Response(200, headers={"content-type": "text/html"}, text="<main>Blog</main>")
        if request.url.path == "/blog/feed/":
            return httpx.Response(200, headers={"content-type": "application/rss+xml"}, text=rss)
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(feed_handler)) as client:
        feed = blogs.fetch_blog(
            client,
            {"url": "https://example.com/blog", "source_mode": "feed"},
        )
    assert feed.method == "rss" and feed.feed_url == "https://example.com/blog/feed/"
    assert calls[:2] == ["/blog", "/blog/feed/"]

    assert blogs._youtube_feed_url(
        "<link rel='alternate' type='application/rss+xml' href='/feeds/videos.xml?channel_id=UCx'>",
        "https://www.youtube.com/x",
    ) == "https://www.youtube.com/feeds/videos.xml?channel_id=UCx"
    assert blogs._youtube_feed_url("<main>No id</main>", "https://youtube.com/x") is None

    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, headers={"content-type": "application/xml"}, text="<not-feed>Readable text</not-feed>")
        )
    ) as client:
        fallback = blogs.fetch_blog(
            client,
            {"url": "https://example.com/page", "source_mode": "article"},
        )
    assert fallback.method == "snapshot" and fallback.snapshot_text == "Readable text"


def test_youtube_uses_ytdlp_video_descriptions(monkeypatch):
    payload = {
        "title": "Neubig videos",
        "entries": [
            {
                "id": "video-1",
                "title": "Lecture one",
                "webpage_url": "https://www.youtube.com/watch?v=video-1",
                "upload_date": "20260731",
                "description": (
                    "The complete lecture description.\n\n"
                    "Slides: https://example.com/slides?utm_source=youtube"
                ),
            }
        ],
    }
    commands: list[list[str]] = []

    monkeypatch.setattr(blogs.importlib.util, "find_spec", lambda name: object())

    def fake_run(command, **kwargs):
        commands.append(command)
        assert kwargs["timeout"] == blogs.YOUTUBE_REQUEST_TIMEOUT
        return blogs.subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(payload),
            stderr="",
        )

    monkeypatch.setattr(blogs.subprocess, "run", fake_run)

    def unexpected_http(request):
        raise AssertionError(f"YouTube HTML should not be fetched: {request.url}")

    with httpx.Client(transport=httpx.MockTransport(unexpected_http)) as client:
        fetched = blogs.fetch_blog(
            client,
            {
                "url": "https://www.youtube.com/neubig/videos",
                "title": "Neubig",
                "source_mode": "video",
            },
        )

    assert commands and commands[0][:3] == [blogs.sys.executable, "-m", "yt_dlp"]
    assert commands[0][-1] == "https://www.youtube.com/neubig/videos"
    assert fetched.method == "rss" and fetched.source_mode == "video"
    assert fetched.feed_items[0].published_at == "2026-07-31"
    assert "complete lecture description" in fetched.feed_items[0].text
    assert fetched.feed_items[0].outbound_links == ("https://example.com/slides",)
    assert "Loaded 1 YouTube video descriptions" in fetched.diagnostics[0]

    monkeypatch.setattr(blogs.importlib.util, "find_spec", lambda name: None)
    with httpx.Client(transport=httpx.MockTransport(unexpected_http)) as client:
        with pytest.raises(ValueError, match="yt-dlp is not installed"):
            blogs.fetch_blog(
                client,
                {
                    "url": "https://www.youtube.com/neubig/videos",
                    "source_mode": "video",
                },
            )


def test_youtube_metadata_and_ytdlp_error_paths(monkeypatch):
    assert blogs._html_first_date(
        "<script>2020-01-01</script><style>x</style><p>Published 2026-08-02</p>"
    ) == "2026-08-02"
    assert blogs._plain_text_html("\n \n") is None
    assert blogs._plain_text_html("One\nline\n\nTwo") == "<p>One<br>line</p><p>Two</p>"

    assert blogs._youtube_published_at({"timestamp": 0}, "") == "1970-01-01T00:00:00+00:00"
    assert blogs._youtube_published_at(
        {"release_timestamp": 10**30, "release_date": "20261340", "upload_date": "Jul 3, 2026"},
        "",
    ) == "2026-07-03"
    assert blogs._youtube_published_at({}, "Recorded on June, 2020") == "2020-06"

    fallback = "https://www.youtube.com/watch?v=fallback"
    item = blogs._youtube_feed_item(
        {"id": "fallback", "url": "https://example.com/not-youtube", "fulltitle": "Fallback"},
        fallback,
    )
    assert item is not None and item.url == fallback and item.published_at is None
    assert "rss-published" not in (item.rendered_html or "")
    assert blogs._youtube_feed_item({}, "not a valid url") is None

    monkeypatch.setattr(blogs.importlib.util, "find_spec", lambda name: object())

    def raises_oserror(*args, **kwargs):
        raise OSError("missing executable")

    monkeypatch.setattr(blogs.subprocess, "run", raises_oserror)
    with pytest.raises(ValueError, match="yt-dlp failed: missing executable"):
        blogs._youtube_fetch({"url": fallback})

    cases = [
        (
            blogs.subprocess.CompletedProcess([], 2, stdout="", stderr="fatal\nlast error"),
            "last error",
        ),
        (blogs.subprocess.CompletedProcess([], 0, stdout="not json", stderr=""), "invalid JSON"),
        (blogs.subprocess.CompletedProcess([], 0, stdout="[]", stderr=""), "invalid result"),
        (
            blogs.subprocess.CompletedProcess([], 0, stdout=json.dumps({"entries": []}), stderr=""),
            "no readable videos",
        ),
    ]
    for process, message in cases:
        monkeypatch.setattr(blogs.subprocess, "run", lambda *args, value=process, **kwargs: value)
        with pytest.raises(ValueError, match=message):
            blogs._youtube_fetch({"url": fallback})

    direct = {
        "id": "direct",
        "title": "Direct video",
        "webpage_url": fallback,
        "description": "Description",
    }
    monkeypatch.setattr(
        blogs.subprocess,
        "run",
        lambda *args, **kwargs: blogs.subprocess.CompletedProcess(
            [], 0, stdout=json.dumps(direct), stderr=""
        ),
    )
    fetched = blogs._youtube_fetch({"url": fallback, "title": "Row title"})
    assert fetched.title == "Direct video" and len(fetched.feed_items) == 1


def test_auto_index_fetches_child_article_and_visible_date():
    index_html = """
    <html><title>Robotics</title><main>
      <article><span>Jul 18, 2022</span>
        <a href='/blog/robotics-post/'>Robotics post</a>
        <p>This is only the index summary.</p>
      </article>
    </main></html>
    """
    article_html = """
    <html><title>Robotics post</title><main>
      <h1>Robotics post</h1>
      <div>ertApr 14, 20265554Share</div>
      <p>The complete article body, not the category index.</p>
      <a href='/blog/nested-reference/'>Nested reference</a>
    </main></html>
    """
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        if request.url.path == "/blog/category/robotics/":
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text=index_html,
            )
        if request.url.path == "/blog/robotics-post/":
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text=article_html,
            )
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        fetched = blogs.fetch_blog(
            client,
            {
                "url": "https://developer.nvidia.com/blog/category/robotics/",
                "title": "Robotics",
                "source_mode": "auto",
                "_known_article_urls": (),
                "article_cursor": 0,
            },
        )

    assert requested == ["/blog/category/robotics/", "/blog/robotics-post/"]
    assert fetched.source_mode == "article_index"
    assert fetched.snapshot_text and "index summary" in fetched.snapshot_text
    assert len(fetched.article_items) == 1
    article = fetched.article_items[0]
    assert article.url == "https://developer.nvidia.com/blog/robotics-post/"
    assert "complete article body" in article.text
    assert article.published_at == "2026-04-14"



def test_auto_direct_article_is_not_misclassified_or_recursively_crawled():
    article_html = """
    <html><title>Main article</title><main>
      <article>
        <h1>Main article</h1>
        <p>The complete source article body.</p>
        <section class='related-posts'>
          <a href='/blog/related-one/'>Related one</a>
          <a href='/blog/related-two/'>Related two</a>
        </section>
      </article>
    </main></html>
    """
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        if request.url.path == "/blog/main-article/":
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text=article_html,
            )
        raise AssertionError(f"Related links must not be crawled: {request.url}")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        fetched = blogs.fetch_blog(
            client,
            {
                "url": "https://example.com/blog/main-article/",
                "title": "Main article",
                "source_mode": "auto",
                "_known_article_urls": (),
                "article_cursor": 0,
            },
        )

    assert requested == ["/blog/main-article/"]
    assert fetched.source_mode == "article"
    assert fetched.article_items == ()
    assert fetched.snapshot_text and "complete source article body" in fetched.snapshot_text
    assert any("lacked index-page evidence" in item for item in fetched.diagnostics)


def test_feed_discovery_and_child_crawling_have_hard_request_caps():
    feed_links = "".join(
        f"<link rel='alternate' type='application/rss+xml' href='/feeds/{index}.xml'>"
        for index in range(20)
    )
    index_html = (
        "<html><head>" + feed_links + "</head><main>"
        + "".join(
            f"<article><a href='/blog/post-{index}/'>Post {index}</a></article>"
            for index in range(20)
        )
        + "</main></html>"
    )
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        if request.url.path == "/blog/":
            return httpx.Response(
                200, headers={"content-type": "text/html"}, text=index_html
            )
        if request.url.path.startswith("/feeds/"):
            return httpx.Response(404)
        if request.url.path.startswith("/blog/post-"):
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text=f"<html><main><h1>{request.url.path}</h1><p>Body</p></main></html>",
            )
        raise AssertionError(request.url)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        fetched = blogs.fetch_blog(
            client,
            {
                "url": "https://example.com/blog/",
                "source_mode": "auto",
                "_known_article_urls": (),
                "article_cursor": 0,
            },
        )

    feed_requests = [path for path in requested if path.startswith("/feeds/")]
    child_requests = [path for path in requested if path.startswith("/blog/post-")]
    assert len(feed_requests) <= blogs.MAX_FEED_CANDIDATES
    assert len(child_requests) <= blogs.MAX_ARTICLE_FETCHES
    assert len(fetched.article_items) <= blogs.MAX_ARTICLE_FETCHES


def test_github_release_commit_and_api_modes():
    empty = "<feed xmlns='http://www.w3.org/2005/Atom'><title>Releases</title></feed>"
    commit = """<feed xmlns='http://www.w3.org/2005/Atom'><title>Commits</title><entry><id>c1</id><title>Update notes</title><link href='https://github.com/org/repo/commit/1'/><updated>2026-08-01T00:00:00Z</updated><summary>Changed docs</summary></entry></feed>"""

    def repo_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/releases.atom"):
            return httpx.Response(200, headers={"content-type": "application/atom+xml"}, text=empty)
        if request.url.path.endswith("/commits.atom"):
            return httpx.Response(200, headers={"content-type": "application/atom+xml"}, text=commit)
        raise AssertionError(request.url)

    with httpx.Client(transport=httpx.MockTransport(repo_handler)) as client:
        repo = blogs.fetch_blog(
            client,
            {"url": "https://github.com/org/repo", "source_mode": "github"},
        )
    assert repo.feed_items[0].title == "Update notes"
    assert any("commits" in item for item in repo.diagnostics)

    payload = [
        None,
        {},
        {
            "html_url": "https://github.com/org/repo",
            "full_name": "org/repo",
            "description": "Repository",
            "language": "Python",
            "archived": False,
            "homepage": "https://project.example/?utm_source=github",
            "created_at": "2026-07-01T00:00:00Z",
        },
    ]

    def org_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/orgs/"):
            return httpx.Response(404)
        if request.url.path.startswith("/users/"):
            return httpx.Response(200, json=payload)
        raise AssertionError(request.url)

    with httpx.Client(transport=httpx.MockTransport(org_handler)) as client:
        org = blogs.fetch_blog(
            client,
            {"url": "https://github.com/org", "source_mode": "github"},
        )
    assert len(org.feed_items) == 1
    assert org.feed_items[0].outbound_links == ("https://project.example/",)
    assert "Language: Python" in org.feed_items[0].text

    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(404))) as client:
        with pytest.raises(ValueError, match="must identify"):
            blogs.fetch_blog(client, {"url": "https://github.com/", "source_mode": "github"})
        with pytest.raises(ValueError, match="not found"):
            blogs.fetch_blog(client, {"url": "https://github.com/missing", "source_mode": "github"})

    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))) as client:
        with pytest.raises(ValueError, match="invalid"):
            blogs.fetch_blog(client, {"url": "https://github.com/org", "source_mode": "github"})


def test_semantic_diff_link_context_and_docs_thresholds():
    assert blogs.blog_diff("334\nfollowers", "333\nfollowers") == ("baseline", "")
    assert blogs.blog_diff(
        "A's profile picture\nB's profile picture",
        "C's profile picture\nD's profile picture",
    ) == ("baseline", "")
    assert blogs.blog_diff("First\nSecond", "Second\nFirst") == ("baseline", "")
    assert blogs.blog_diff("Alpha beta\nGamma", "Gamma\nAlpha beta") == ("baseline", "")

    previous = "Body\nOutbound links:\n- Source -> https://example.com/old"
    current = "Body\nOutbound links:\n- Source -> https://example.com/new"
    kind, diff = blogs.blog_diff(previous, current, source_mode="article")
    assert kind == "changed" and "+- Source -> https://example.com/new" in diff

    assert blogs.blog_diff("Body", current) == ("baseline", "")
    one_added = "Body\nOutbound links:\n- One -> https://example.com/one"
    assert blogs.blog_diff("Body\nOutbound links:", one_added) == ("baseline", "")
    two_added = one_added + "\n- Two -> https://example.com/two"
    assert blogs.blog_diff("Body\nOutbound links:", two_added)[0] == "changed"
    assert blogs.blog_diff(
        "Body\nOutbound links:\n- A -> https://x/a\n- B -> https://x/b",
        "Body\nOutbound links:\n- B -> https://x/b\n- A -> https://x/a",
    ) == ("baseline", "")

    small_docs_old = "\n".join(f"Line {i}" for i in range(30))
    small_docs_new = small_docs_old.replace("Line 5", "Changed 5")
    assert blogs.blog_diff(small_docs_old, small_docs_new, source_mode="docs") == (
        "baseline",
        "",
    )
    large_docs_new = "\n".join(f"Replacement {i}" for i in range(30))
    assert blogs.blog_diff(small_docs_old, large_docs_new, source_mode="docs")[0] == "reset"
    assert blogs._semantic_line("  ") == ""
    assert blogs._semantic_line("12,345") == ""
    assert blogs._semantic_line("Followers 123") == "followers #"
    assert blogs._link_sets(None) is None
    assert blogs._split_snapshot("URL: https://example.com/one")[1] == {
        "__direct__": frozenset({"https://example.com/one"})
    }


def test_database_source_settings_known_articles_and_schema(app_config, author_candidate):
    with Database(app_config.paths.database) as db:
        assert SCHEMA_VERSION == 34
        db.add_author(author_candidate)
        blog_id = db.add_author_blog(
            "A123",
            "https://example.com/blog",
            source_mode="article_index",
            content_selector="main",
            article_url_pattern=r"/posts/",
        )
        row = db.author_blogs("A123")[0]
        assert row["source_mode"] == "article_index"
        assert row["content_selector"] == "main"
        with pytest.raises(ValueError, match="Unknown"):
            db.add_author_blog("A123", "https://example.com/bad", source_mode="bad")

        new = db.sync_known_blog_articles(
            blog_id,
            [
                ("https://example.com/posts/one", "One"),
                ("https://example.com/posts/two", "Two"),
            ],
            fetched_urls=["https://example.com/posts/one"],
        )
        assert new == {
            "https://example.com/posts/one",
            "https://example.com/posts/two",
        }
        assert db.known_blog_article_urls(blog_id) == (
            "https://example.com/posts/two",
            "https://example.com/posts/one",
        )
        assert db.sync_known_blog_articles(
            blog_id,
            [("https://example.com/posts/two", "Two updated")],
            fetched_urls=["https://example.com/posts/one"],
        ) == set()

        assert db.update_author_blog_url(
            "A123",
            blog_id,
            "https://example.com/blog",
            source_mode="docs",
            content_selector=".docs",
            article_url_pattern="",
        )
        configured = db.author_blogs("A123")[0]
        assert configured["source_mode"] == "docs"
        assert configured["content_selector"] == ".docs"
        assert configured["article_url_pattern"] is None
        db.mark_blog_checked(
            blog_id,
            status=200,
            fetch_method="snapshot",
            article_initialized=True,
            article_cursor=3,
            diagnostics=json.dumps(["selected"]),
            item_stream_mode="article_index",
        )
        checked = db.author_blogs("A123")[0]
        assert checked["article_initialized"] == 1
        assert checked["article_cursor"] == 3
        assert checked["item_stream_mode"] == "article_index"
        assert db.update_author_blog_url("A123", blog_id, "https://example.com/new")
        changed = db.author_blogs("A123")[0]
        assert changed["source_mode"] == "docs"
        assert db.known_blog_article_urls(blog_id) == ()


def test_pipeline_article_discovery_and_child_updates(app_config, author_candidate):
    with Database(app_config.paths.database) as db:
        db.add_author(author_candidate)
        blog_id = db.add_author_blog(
            "A123",
            "https://example.com/blog",
            source_mode="article_index",
        )
        app = PaperWatch(
            app_config,
            db=db,
            openalex=NoPapers(),
            arxiv=None,
            llm=NoLLM(),
            pdf_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500))),
        )
        legacy_id = db.save_blog_entry(
            blog_id=blog_id,
            title="Index",
            content_sha256=hashlib.sha256(b"old index").hexdigest(),
            normalized_text="Old index page",
            diff_text="+Old index page",
            change_kind="changed",
            previous_entry_id=None,
            entry_kind="snapshot",
            external_id="source-page",
            source_url="https://example.com/blog",
        )
        first_row = app._prepare_blog_fetch_rows(db.list_enabled_blogs())[0]
        first_text = "Title: One\nURL: https://example.com/posts/one\nInitial body"
        first = BlogFetch(
            method="snapshot",
            title="Index",
            status_code=200,
            snapshot_text="Index one",
            snapshot_sha256=hashlib.sha256(b"Index one").hexdigest(),
            source_mode="article_index",
            diagnostics=("initial",),
            article_cursor_next=1,
            discovered_articles=(
                ArticleLink("One", "https://example.com/posts/one", "Summary one"),
            ),
            article_items=(
                ArticleItem(
                    external_id=hashlib.sha256(b"body-one").hexdigest(),
                    title="One",
                    url="https://example.com/posts/one",
                    text=first_text,
                    sha256=hashlib.sha256(first_text.encode()).hexdigest(),
                    published_at="2024-03-12",
                ),
            ),
        )
        first_stats = RunStats(run_id=db.start_run())
        app._apply_blog_fetch(first_stats, first_row, first, None)
        assert first_stats.blog_updates_found == 0
        assert db.known_blog_article_urls(blog_id) == ("https://example.com/posts/one",)
        initial_entries = db.list_blog_entries()
        assert len(initial_entries) == 1
        assert initial_entries[0]["title"] == "One"
        assert initial_entries[0]["source_url"] == "https://example.com/posts/one"
        assert initial_entries[0]["published_at"] == "2024-03-12"
        assert db.get_blog_entry(legacy_id)["change_kind"] == "baseline"
        assert db.get_blog_entry(legacy_id)["diff_text"] == ""

        second_row = app._prepare_blog_fetch_rows(db.list_enabled_blogs())[0]
        second_text = "Title: One\nURL: https://example.com/posts/one\nSubstantially changed body"
        second = BlogFetch(
            method="snapshot",
            title="Index",
            status_code=200,
            snapshot_text="Index one\nIndex two",
            snapshot_sha256=hashlib.sha256(b"Index one\nIndex two").hexdigest(),
            source_mode="article_index",
            diagnostics=("second",),
            article_cursor_next=2,
            discovered_articles=(
                ArticleLink("One", "https://example.com/posts/one", "Summary one"),
                ArticleLink("Two", "https://example.com/posts/two", "Summary two"),
            ),
            article_items=(
                ArticleItem(
                    external_id=hashlib.sha256(b"body-one").hexdigest(),
                    title="One",
                    url="https://example.com/posts/one",
                    text=second_text,
                    sha256=hashlib.sha256(second_text.encode()).hexdigest(),
                ),
            ),
        )
        second_stats = RunStats(run_id=db.start_run())
        app._apply_blog_fetch(second_stats, second_row, second, None)
        assert second_stats.blog_updates_found == 1
        updates = db.blog_entries_by_ids(second_stats.new_blog_entry_ids)
        assert {row["source_url"] for row in updates} == {
            "https://example.com/posts/one",
        }
        assert all(
            row["source_url"] != "https://example.com/blog"
            for row in db.list_blog_entries()
        )
        assert set(db.known_blog_article_urls(blog_id)) == {
            "https://example.com/posts/one",
            "https://example.com/posts/two",
        }
        stored = db.author_blogs("A123")[0]
        assert stored["article_cursor"] == 2
        assert "second" in stored["feed_diagnostics"]

        current = db.latest_blog_entry(
            blog_id, entry_kind="rss", external_id=hashlib.sha256(b"body-one").hexdigest()
        )
        assert current is not None
        with db.transaction() as conn:
            conn.execute("UPDATE blog_entries SET published_at=NULL WHERE id=?", (current["id"],))
        unchanged_with_date = BlogFetch(
            method="snapshot",
            title="Index",
            status_code=200,
            snapshot_text="Index one\nIndex two",
            snapshot_sha256=hashlib.sha256(b"Index one\nIndex two").hexdigest(),
            source_mode="article_index",
            article_cursor_next=2,
            discovered_articles=second.discovered_articles,
            article_items=(
                ArticleItem(
                    external_id=hashlib.sha256(b"body-one").hexdigest(),
                    title="One",
                    url="https://example.com/posts/one",
                    text=second_text,
                    sha256=hashlib.sha256(second_text.encode()).hexdigest(),
                    published_at="2026-04-14",
                ),
            ),
        )
        third_stats = RunStats(run_id=db.start_run())
        third_row = app._prepare_blog_fetch_rows(db.list_enabled_blogs())[0]
        app._apply_blog_fetch(third_stats, third_row, unchanged_with_date, None)
        assert third_stats.blog_updates_found == 0
        assert db.get_blog_entry(int(current["id"]))["published_at"] == "2026-04-14"
        app.close()


def test_initial_publications_are_visible_without_being_reported_as_updates(
    app_config, author_candidate
):
    with Database(app_config.paths.database) as db:
        db.add_author(author_candidate)
        db.add_author_blog(
            "A123",
            "https://example.com/publications",
            source_mode="publications",
        )
        app = PaperWatch(
            app_config,
            db=db,
            openalex=NoPapers(),
            arxiv=None,
            llm=NoLLM(),
            pdf_client=httpx.Client(
                transport=httpx.MockTransport(lambda request: httpx.Response(500))
            ),
        )
        row = app._prepare_blog_fetch_rows(db.list_enabled_blogs())[0]
        fetched = BlogFetch(
            method="snapshot",
            title="Publications",
            status_code=200,
            snapshot_text="Publication index",
            snapshot_sha256=hashlib.sha256(b"Publication index").hexdigest(),
            source_mode="publications",
            discovered_articles=(
                ArticleLink("Paper one", "https://doi.org/10.1/one", "First paper"),
                ArticleLink("Paper two", "https://doi.org/10.1/two", "Second paper"),
            ),
        )
        stats = RunStats(run_id=db.start_run())
        app._apply_blog_fetch(stats, row, fetched, None)
        assert stats.blog_updates_found == 0
        assert len(stats.new_blog_entry_ids) == 2
        assert {entry["title"] for entry in db.list_blog_entries()} == {
            "Paper one",
            "Paper two",
        }
        app.close()


def test_existing_publication_baseline_is_recovered_after_upgrade(
    app_config, author_candidate
):
    with Database(app_config.paths.database) as db:
        db.add_author(author_candidate)
        blog_id = db.add_author_blog(
            "A123",
            "https://example.com/publications",
            source_mode="publications",
        )
        db.sync_known_blog_articles(
            blog_id,
            [("https://doi.org/10.1/old", "Older paper")],
        )
        db.mark_blog_checked(
            blog_id,
            status=200,
            fetch_method="snapshot",
            article_initialized=True,
            item_stream_mode="publications",
        )
        app = PaperWatch(
            app_config,
            db=db,
            openalex=NoPapers(),
            arxiv=None,
            llm=NoLLM(),
            pdf_client=httpx.Client(
                transport=httpx.MockTransport(lambda request: httpx.Response(500))
            ),
        )
        row = app._prepare_blog_fetch_rows(db.list_enabled_blogs())[0]
        fetched = BlogFetch(
            method="snapshot",
            title="Publications",
            status_code=200,
            snapshot_text="Publication index",
            snapshot_sha256=hashlib.sha256(b"Publication index").hexdigest(),
            source_mode="publications",
            discovered_articles=(
                ArticleLink("Older paper", "https://doi.org/10.1/old", "Stored before upgrade"),
            ),
        )
        stats = RunStats(run_id=db.start_run())
        app._apply_blog_fetch(stats, row, fetched, None)
        assert stats.blog_updates_found == 0
        entries = db.list_blog_entries()
        assert len(entries) == 1
        assert entries[0]["title"] == "Older paper"
        app.close()


def test_blog_fetch_batch_fetches_multiple_sources_concurrently(
    app_config, monkeypatch
):
    barrier = threading.Barrier(2)

    def fake_fetch(client, row):
        barrier.wait(timeout=2)
        return BlogFetch(
            method="snapshot",
            title=str(row["url"]),
            status_code=200,
            snapshot_text="body",
            snapshot_sha256="hash",
        )

    monkeypatch.setattr("paperwatch.pipeline.fetch_blog", fake_fetch)
    with Database(app_config.paths.database) as db:
        app = PaperWatch(
            app_config,
            db=db,
            openalex=NoPapers(),
            arxiv=None,
            llm=NoLLM(),
            pdf_client=httpx.Client(
                transport=httpx.MockTransport(lambda request: httpx.Response(500))
            ),
        )
        rows = [
            {"id": 1, "url": "https://one.example/blog"},
            {"id": 2, "url": "https://two.example/blog"},
        ]
        results = app._fetch_blog_batch(rows)
        assert [result[0]["id"] for result in results] == [1, 2]
        assert all(result[1] is not None and result[2] is None for result in results)
        app.close()



def test_known_article_history_is_bounded(app_config, author_candidate, monkeypatch):
    monkeypatch.setattr(db_module, "MAX_KNOWN_BLOG_ARTICLES", 2)
    with Database(app_config.paths.database) as db:
        db.add_author(author_candidate)
        blog_id = db.add_author_blog("A123", "https://example.com/blog")
        db.sync_known_blog_articles(
            blog_id,
            [(f"https://example.com/posts/{index}", str(index)) for index in range(5)],
        )
        db.sync_known_blog_articles(
            blog_id,
            [("https://example.com/posts/current", "Current")],
        )
        rows = db._conn.execute(
            "SELECT article_url, active FROM known_blog_articles WHERE blog_id=?",
            (blog_id,),
        ).fetchall()
        assert len(rows) == 3
        assert sum(int(row["active"]) for row in rows) == 1
        known = db.known_blog_article_urls(blog_id)
        assert known[0] == "https://example.com/posts/current"
        assert len(known) == 2


def test_schema_18_promotes_and_seeds_initial_feed_items(tmp_path, author_candidate):
    path = tmp_path / "blog-migration.sqlite"
    with Database(path) as db:
        db.add_author(author_candidate)
        blog_id = db.add_author_blog("A123", "https://example.com/feed")
        db.save_blog_entry(
            blog_id=blog_id,
            title="Existing post",
            content_sha256="hash",
            normalized_text="Existing body",
            diff_text="",
            change_kind="baseline",
            previous_entry_id=None,
            entry_kind="rss",
            external_id="existing",
            source_url="https://example.com/post",
        )
        db._conn.execute("PRAGMA user_version = 16")
        db._conn.commit()

    with Database(path) as db:
        entries = db.list_blog_entries()
        assert len(entries) == 1
        assert entries[0]["change_kind"] == "changed"
        assert db._conn.execute("PRAGMA user_version").fetchone()[0] == 34
        assert db.known_blog_article_urls(blog_id) == ("https://example.com/post",)


def test_unchanged_feed_still_rechecks_known_article_pages():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/feed.xml":
            return httpx.Response(304, request=request)
        if request.url.path == "/posts/one":
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="<title>One</title><main>Website changed independently</main>",
                request=request,
            )
        raise AssertionError(request.url)

    row = {
        "url": "https://example.com/blog",
        "title": "Blog",
        "source_mode": "auto",
        "feed_url": "https://example.com/feed.xml",
        "feed_etag": "v1",
        "feed_discovery_checked": 1,
        "_known_article_urls": ("https://example.com/posts/one",),
        "article_cursor": 0,
    }
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        fetched = blogs.fetch_blog(client, row)
    assert fetched.method == "rss"
    assert not fetched.not_modified
    assert fetched.feed_items == ()
    assert len(fetched.article_items) == 1
    assert "Website changed independently" in fetched.article_items[0].text
    assert seen == ["/feed.xml", "/posts/one"]


def test_rss_pipeline_stores_linked_page_and_tracks_later_page_change(
    app_config, author_candidate
):
    with Database(app_config.paths.database) as db:
        db.add_author(author_candidate)
        db.add_author_blog("A123", "https://example.com/blog")
        app = PaperWatch(
            app_config,
            db=db,
            openalex=NoPapers(),
            arxiv=None,
            llm=NoLLM(),
        )
        first_row = app._prepare_blog_fetch_rows(db.list_enabled_blogs())[0]
        feed_text = "Title: One\nURL: https://example.com/posts/one\nFeed summary"
        page_text = "Title: One\nURL: https://example.com/posts/one\nFull website body"
        external_id = hashlib.sha256(b"feed-one").hexdigest()
        article_id = hashlib.sha256(b"article-one").hexdigest()
        first = BlogFetch(
            method="rss",
            title="Feed",
            status_code=200,
            feed_url="https://example.com/feed.xml",
            feed_items=(
                blogs.FeedItem(
                    external_id=external_id,
                    title="One",
                    url="https://example.com/posts/one",
                    published_at=None,
                    text=feed_text,
                    sha256=hashlib.sha256(feed_text.encode()).hexdigest(),
                ),
            ),
            discovered_articles=(
                ArticleLink("One", "https://example.com/posts/one", "Feed summary"),
            ),
            article_items=(
                ArticleItem(
                    external_id=article_id,
                    title="One",
                    url="https://example.com/posts/one",
                    text=page_text,
                    sha256=hashlib.sha256(page_text.encode()).hexdigest(),
                ),
            ),
            source_mode="auto",
            article_cursor_next=1,
        )
        first_stats = RunStats(run_id=db.start_run())
        app._apply_blog_fetch(first_stats, first_row, first, None)
        assert first_stats.blog_updates_found == 0
        entries = db.list_blog_entries()
        assert len(entries) == 1
        assert {entry["entry_kind"] for entry in entries} == {"rss"}
        assert "Full website body" in entries[0]["normalized_text"]

        second_row = app._prepare_blog_fetch_rows(db.list_enabled_blogs())[0]
        changed_page = page_text.replace("Full website body", "Substantially changed website body")
        second = BlogFetch(
            method="rss",
            title="Feed",
            status_code=304,
            feed_url="https://example.com/feed.xml",
            article_items=(
                ArticleItem(
                    external_id=article_id,
                    title="One",
                    url="https://example.com/posts/one",
                    text=changed_page,
                    sha256=hashlib.sha256(changed_page.encode()).hexdigest(),
                ),
            ),
            source_mode="auto",
            article_cursor_next=2,
        )
        second_stats = RunStats(run_id=db.start_run())
        app._apply_blog_fetch(second_stats, second_row, second, None)
        assert second_stats.blog_updates_found == 1
        changed = db.blog_entries_by_ids(second_stats.new_blog_entry_ids)[0]
        assert changed["entry_kind"] == "rss"
        assert changed["source_url"] == "https://example.com/posts/one"
        assert "Substantially changed" in changed["diff_text"]
        app.close()
